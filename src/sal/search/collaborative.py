#!/usr/bin/env python
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from vllm import SamplingParams

from sal.config import Config
from sal.utils.math import extract_completion_answers, memoized_canonical_form
from typing import List
import json
import random
from collections import Counter
from concurrent.futures import ThreadPoolExecutor


def get_resp(config, llm_big, conv, temperature):
    resp = llm_big.chat.completions.create(
        model=config.model_path_big,
        temperature=temperature,
        max_tokens=config.max_tokens,
        messages=conv,
        # cs470: DeepSeek V4.1 Flash thinks by default; the proposal pilot disabled it.
        extra_body=None if config.big_thinking else {"thinking": {"type": "disabled"}},
    )
    return resp


def extract_preds_from_compl(completions_group: List[List[str]], guides_group: List[List[str]], thresh):
    preds_selected = []
    guides_selected = []
    votes = []  # cs470: the vote behind each decision, for the stage log

    for completions, guides in zip(completions_group, guides_group):
        answers = extract_completion_answers({"completions": completions})["preds"]
        answers = [memoized_canonical_form(ans) for ans in answers]

        counts = Counter(answers)
        majority_answer, count = counts.most_common(1)[0]
        ratio = count / len(answers)

        if ratio > thresh:
            candidates = [(i, comp, guide) for i, (comp, guide, ans) in enumerate(zip(completions, guides, answers)) if ans == majority_answer]
            chosen, pred_selected, guide_selected = random.choice(candidates)
        else:
            chosen, pred_selected, guide_selected = None, None, None

        preds_selected.append(pred_selected)
        guides_selected.append(guide_selected)
        votes.append({"answers": answers, "counts": counts.most_common(), "majority": majority_answer,
                      "share": ratio, "threshold": thresh, "accepted": ratio > thresh, "chosen": chosen})

    return preds_selected, guides_selected, votes


def get_active_idxs_and_probs(items):
    idxs_active = [i for i, item in enumerate(items) if item["pred"] is None]
    probs_active = [items[i]["problem"] for i in idxs_active]
    return idxs_active, probs_active


def count_tokens_vllm(templated_convs, responses, n, tokenizer):
    counts_in  = []
    counts_out = []

    for i in range(len(templated_convs)):
        tok_in = len(tokenizer.encode(templated_convs[i]))
        tok_out = len(responses[i].outputs[0].token_ids)
        counts_in.append(tok_in)
        counts_out.append(tok_out)

    merged_in  = [sum(counts_in[i*n:(i+1)*n])  for i in range(len(counts_in)//n)]
    merged_out = [sum(counts_out[i*n:(i+1)*n]) for i in range(len(counts_out)//n)]
    return merged_in, merged_out


def _string_or_empty(value):
    return value if value is not None else ""


def finalize_outputs(x, items):
    preds = [_string_or_empty(item["pred"]) for item in items]
    # cs470: every stage's prompts, outputs and votes, as a JSON string per task
    # (a string keeps the dataset schema fixed whichever stages ran).
    x["stages"] = [json.dumps(item["stages"]) for item in items]

    x["guide_big"] = [_string_or_empty(item["guide_big"]) for item in items]
    x["guides_small"] = [
        [_string_or_empty(guide) for guide in item["guides_small"]]
        for item in items
    ]
    x["guide_small"] = [_string_or_empty(item["guide_small"]) for item in items]
    x["pred"] = preds
    x["completions"] = [[pred] for pred in preds]
    x["scores"] = [[[0.0]] for _ in preds]

    x["small_in"] = [int(item["small_in"]) for item in items]
    x["small_out"] = [int(item["small_out"]) for item in items]
    x["big_in"] = [int(item["big_in"]) for item in items]
    x["big_out"] = [int(item["big_out"]) for item in items]
    x["round"] = [int(item["round"] or 0) for item in items]
    return x


def generate_big(convs, config, llm_big, idxs, items, temperature, target, stage):
    with ThreadPoolExecutor(max_workers=config.max_workers) as executor:
        responses = list(executor.map(lambda conv: get_resp(config, llm_big, conv, temperature), convs))

    for idx, conv, resp in zip(idxs, convs, responses):
        items[idx]["big_in"]  += resp.usage.prompt_tokens
        items[idx]["big_out"] += resp.usage.completion_tokens

        content = resp.choices[0].message.content
        items[idx]["stages"].append({
            "stage": stage, "model": "big", "temperature": temperature, "messages": [conv],
            "outputs": [content], "finish_reasons": [resp.choices[0].finish_reason],
            "output_tokens": [resp.usage.completion_tokens],
        })

        if target == "guide":
            items[idx]["guide_big"] = content
        elif target == "pred":
            items[idx]["pred"] = content

    return items


def clean_output(text: str) -> str:
    stop_strings = ['<|eot_id|>', '<|endoftext|>', '<|im_end|>']
    for stop in stop_strings:
        if stop in text:
            text = text.split(stop)[0]
    return text.strip()


def generate_small(convs, config, llm_small, idxs, items, temperature, target, duplicate, stage, thresh=None):
    tokenizer = llm_small.get_tokenizer()

    if config.custom_chat_template is not None:
        tokenizer.chat_template = config.custom_chat_template

    templated_convs = tokenizer.apply_chat_template(
        convs, tokenize=False, add_generation_prompt=True, **config.chat_template_kwargs
    )

    if duplicate:
        # Duplicate convs to generate config.n completions per prompt so we can do continous batching
        # This makes [p1, p2, p3, p4] become [p1, p1, p2, p2, p3, p3, p4, p4] for e.g. config.n=2
        templated_convs = [c for conv in templated_convs for c in [conv] * config.n]

    prefix = "The goal"

    if target == "guide":
        templated_convs = [templ_conv + prefix for templ_conv in templated_convs]

        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=config.max_tokens,
            top_p=config.top_p,
            n=1,  # Since we've already duplicated the prompt_token_ids, we only need to generate 1 completion per prompt
            skip_special_tokens=True,
            stop=['. ', '\n'],
            include_stop_str_in_output=True,
        )
    elif target == "pred":
        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=config.max_tokens,
            top_p=config.top_p,
            n=1,  # Since we've already duplicated the prompt_token_ids, we only need to generate 1 completion per prompt
            skip_special_tokens=True,
        )

    responses = llm_small.generate(
        templated_convs,
        sampling_params=sampling_params,
        use_tqdm=False,
    )

    completions_group = [[] for _ in range(len(idxs))]

    if len(responses) == len(idxs) * config.n:
        if target == "guide":
            for i in range(len(completions_group)):
                completions_group[i] = [
                    clean_output(prefix + output.text)
                    for r in responses[i * config.n : (i + 1) * config.n]
                    for output in r.outputs
                ]
        elif target == "pred":
            for i in range(len(completions_group)):
                completions_group[i] = [
                    clean_output(output.text)
                    for r in responses[i * config.n : (i + 1) * config.n]
                    for output in r.outputs
                ]
    elif target == "pred":
        for i in range(len(completions_group)):
            completions_group[i] = [
                clean_output(output.text)
                for r in responses[i : (i + 1)]
                for output in r.outputs
            ]

    # cs470: per-task slices of the prompts and responses, for the stage log.
    per_task = len(responses) // len(idxs)
    for i, idx in enumerate(idxs):
        group = responses[i * per_task:(i + 1) * per_task]
        prompts = convs if duplicate else convs[i * per_task:(i + 1) * per_task]
        items[idx]["stages"].append({
            "stage": stage, "model": "small", "temperature": temperature,
            # one conversation if every sample shares it, else one per sample
            "messages": [prompts[i]] if duplicate else prompts,
            "outputs": completions_group[i],
            "finish_reasons": [o.finish_reason for r in group for o in r.outputs],
            "output_tokens": [len(o.token_ids) for r in group for o in r.outputs],
        })

    if target == "guide":
        for idx, guides in zip(idxs, completions_group):
            items[idx]["guides_small"] = guides
    elif target == "pred":
        guides_group = [items[idx]["guides_small"] for idx in idxs]
        preds, guides, votes = extract_preds_from_compl(completions_group, guides_group, thresh)

        for idx, pred, guide, vote in zip(idxs, preds, guides, votes):
            items[idx]["pred"] = pred
            items[idx]["guide_small"] = guide
            items[idx]["stages"][-1]["vote"] = vote

    small_in, small_out = count_tokens_vllm(templated_convs, responses, config.n, tokenizer)

    for idx, (tok_in, tok_out) in zip(idxs, zip(small_in, small_out)):
        items[idx]["small_in"]  += tok_in
        items[idx]["small_out"] += tok_out

    return items


def collaborative(x, config: Config, llm_big, llm_small):
    items = [{"problem": problem, "pred": None, "guide_big": None, "guides_small": [""] * config.n, "guide_small": None, "small_in": 0, "small_out": 0, "big_in": 0, "big_out": 0, "round": None, "stages": []} for problem in x["problem"]]

    #################### ROUND 1 ###############################################################################
    idxs_active, probs_active = get_active_idxs_and_probs(items)

    convs_active = [
        [
            {"role": "system", "content": config.system_prompt_guide_small},
            {"role": "user", "content": prob},
        ]
        for prob in probs_active
    ]

    items = generate_small(convs_active, config, llm_small, idxs_active, items, temperature=0.8, target="guide", duplicate=True, stage="1a")

    guides_small_active = [items[idx]["guides_small"] for idx in idxs_active]

    convs_active = []

    for prob, guides_small in zip(probs_active, guides_small_active):
        for guide_small in guides_small:
            convs_active.append([
                {"role": "system", "content": config.system_prompt},
                {"role": "user", "content": config.user_prompt_guide_1.format(problem=prob, guide_small=guide_small)},
            ])

    items = generate_small(convs_active, config, llm_small, idxs_active, items, temperature=0.4, target="pred", duplicate=False, stage="1b", thresh=0.75)

    for idx in idxs_active:
        if items[idx]["pred"] is not None:
            items[idx]["round"] = 1

    if all(item["pred"] is not None for item in items):
        x = finalize_outputs(x, items)
        return x

    #################### ROUND 2 ###############################################################################
    idxs_active, probs_active = get_active_idxs_and_probs(items)

    convs_active = [
        [
            {"role": "system", "content": config.system_prompt_guide_big},
            {"role": "user", "content": prob},
        ]
        for prob in probs_active
    ]

    items = generate_big(convs_active, config, llm_big, idxs_active, items, temperature=0.8, target="guide", stage="2a")

    guides_big_active = [items[idx]["guide_big"] for idx in idxs_active]
    guides_small_active = [items[idx]["guide_small"] for idx in idxs_active]

    convs_active = [
        [
            {"role": "system", "content": config.system_prompt},
            {"role": "user", "content": config.user_prompt_guide_2.format(problem=prob, guide_small=guide_small, guide_big=guide_big)},
        ]
        for prob, guide_small, guide_big in zip(probs_active, guides_small_active, guides_big_active)
    ]

    items = generate_small(convs_active, config, llm_small, idxs_active, items, temperature=0.8, target="pred", duplicate=True, stage="2b", thresh=0.5)

    for idx in idxs_active:
        if items[idx]["pred"] is not None:
            items[idx]["round"] = 2

    if all(item["pred"] is not None for item in items):
        x = finalize_outputs(x, items)
        return x

    #################### ROUND 3 ###############################################################################
    idxs_active, probs_active = get_active_idxs_and_probs(items)

    convs_active = [
        [
            {"role": "system", "content": config.system_prompt},
            {"role": "user", "content": config.user_prompt_guide_3.format(problem=prob, guide_big=items[idx]["guide_big"])},
        ]
        for idx, prob in zip(idxs_active, probs_active)
    ]

    items = generate_big(convs_active, config, llm_big, idxs_active, items, temperature=0, target="pred", stage="3")

    for idx in idxs_active:
        if items[idx]["pred"] is not None:
            items[idx]["round"] = 3

    x = finalize_outputs(x, items)
    return x


def big_guide(x, config: Config, llm_big, llm_small):
    """cs470: COPE's stage 2 for every task. The large model writes the guideline and the
    small model solves n times with it; nothing is escalated. Compared with stage 1b of the
    cascade (the small model's own goal, every task), this measures the guideline's author
    without the cascade's selection of hard tasks. As in the cascade, a task reaches stage 2
    with no accepted small goal, so the goal is passed as None.
    """
    items = [{"problem": problem, "pred": None, "guide_big": None, "guides_small": [""] * config.n, "guide_small": None, "small_in": 0, "small_out": 0, "big_in": 0, "big_out": 0, "round": 2, "stages": []} for problem in x["problem"]]
    idxs, probs = get_active_idxs_and_probs(items)

    convs = [
        [
            {"role": "system", "content": config.system_prompt_guide_big},
            {"role": "user", "content": prob},
        ]
        for prob in probs
    ]
    items = generate_big(convs, config, llm_big, idxs, items, temperature=0.8, target="guide", stage="2a")

    convs = [
        [
            {"role": "system", "content": config.system_prompt},
            {"role": "user", "content": config.user_prompt_guide_2.format(problem=prob, guide_small=None, guide_big=items[idx]["guide_big"])},
        ]
        for idx, prob in zip(idxs, probs)
    ]
    items = generate_small(convs, config, llm_small, idxs, items, temperature=0.8, target="pred", duplicate=True, stage="2b", thresh=0.5)

    # No stage 3 here: a rejected vote still submits an attempt with the most common answer.
    for item in items:
        if item["pred"] is None:
            stage = item["stages"][-1]
            item["pred"] = stage["outputs"][stage["vote"]["answers"].index(stage["vote"]["majority"])]

    return finalize_outputs(x, items)
