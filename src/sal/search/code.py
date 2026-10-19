# cs470: COPE for code generation (MBPP), reimplemented from the paper.
#
# The released code only covers maths. The paper (Lee et al., TMLR 2026, Sec. 4
# and 5.2) describes the code variant: the same three stages as maths, but a
# stage accepts if at least one candidate passes the task's public tests (a
# passing candidate is chosen at random); otherwise the task escalates.
# Final correctness uses the hidden MBPP-ET tests. This module mirrors
# collaborative.py's stages, temperatures and prompt templates.

import random
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from vllm import SamplingParams

from sal.search.collaborative import clean_output, get_resp

# The base (Nix store) interpreter, not the venv: only /nix is visible in the
# sandbox, so generated code gets the standard library and nothing else.
PYTHON = str(Path(sys.base_prefix) / "bin" / "python3")
BWRAP = shutil.which("bwrap")


# --- running code --------------------------------------------------------------
def extract_code(text):
    """The code blocks that define something, in order.

    Answers often add usage or test blocks after the solution; those are
    dropped. Falls back to all blocks, then to the whole text.
    """
    blocks = re.findall(r"```(?:python|py)?[ \t]*\n(.*?)```", text, re.S)
    defining = [b for b in blocks if re.search(r"^\s*(def|class|import|from)\s", b, re.M)]
    return "\n".join(defining or blocks) if blocks else text


def passes(code, tests, setup="", timeout=10):
    """True if `code` passes every assert in `tests`, run isolated (no network)."""
    program = "\n".join([setup, code, *tests]) + "\n"
    with tempfile.TemporaryDirectory() as tmp:
        Path(tmp, "prog.py").write_text(program)
        cmd = [BWRAP, "--ro-bind", "/nix", "/nix", "--dev", "/dev", "--proc", "/proc",
               "--tmpfs", "/tmp", "--bind", tmp, "/work", "--chdir", "/work",
               "--unshare-all", "--die-with-parent", PYTHON, "-I", "prog.py"]
        try:
            return subprocess.run(cmd, capture_output=True, timeout=timeout).returncode == 0
        except subprocess.TimeoutExpired:
            return False


# --- model calls ---------------------------------------------------------------
def small_generate(llm_small, config, convs, temperature, guide=False):
    """One completion per conversation; returns texts, (in, out) token counts and finish reasons."""
    tokenizer = llm_small.get_tokenizer()
    prompts = tokenizer.apply_chat_template(convs, tokenize=False, add_generation_prompt=True)
    prefix = "The goal"
    if guide:
        prompts = [p + prefix for p in prompts]
        params = SamplingParams(temperature=temperature, max_tokens=config.max_tokens,
                                top_p=config.top_p, n=1, skip_special_tokens=True,
                                stop=[". ", "\n"], include_stop_str_in_output=True)
    else:
        params = SamplingParams(temperature=temperature, max_tokens=config.max_tokens,
                                top_p=config.top_p, n=1, skip_special_tokens=True)
    out = llm_small.generate(prompts, sampling_params=params, use_tqdm=False)
    texts = [clean_output((prefix if guide else "") + o.outputs[0].text) for o in out]
    tokens = [(len(o.prompt_token_ids), len(o.outputs[0].token_ids)) for o in out]
    return texts, tokens, [o.outputs[0].finish_reason for o in out]


def big_generate(llm_big, config, conv, temperature):
    resp = get_resp(config, llm_big, conv, temperature)
    return resp.choices[0].message.content or "", \
        (resp.usage.prompt_tokens, resp.usage.completion_tokens), resp.choices[0].finish_reason


def conv(system, user):
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


# --- approaches ---------------------------------------------------------------
def cope_code(task, config, llm_big, llm_small, rng):
    """Run COPE's three stages on one MBPP task. Returns the record for the task.

    cs470: rec["stages"] logs every stage's prompts, outputs, finish reasons and,
    for the solving stages, which candidates passed the public tests.
    """
    rec = {"small_in": 0, "small_out": 0, "big_in": 0, "big_out": 0,
           "guide_small": None, "guide_big": None, "stages": []}
    problem, public = task["problem"], task["public_tests"]

    def small(stage, convs, temperature, guide=False):
        texts, tokens, finish = small_generate(llm_small, config, convs, temperature, guide)
        rec["small_in"] += sum(t[0] for t in tokens)
        rec["small_out"] += sum(t[1] for t in tokens)
        same = all(c == convs[0] for c in convs)
        rec["stages"].append({"stage": stage, "model": "small", "temperature": temperature,
                              "messages": [convs[0]] if same else convs, "outputs": texts,
                              "finish_reasons": finish, "output_tokens": [t[1] for t in tokens]})
        return texts

    def big(stage, c, temperature):
        text, tok, finish = big_generate(llm_big, config, c, temperature)
        rec["big_in"] += tok[0]; rec["big_out"] += tok[1]
        rec["stages"].append({"stage": stage, "model": "big", "temperature": temperature,
                              "messages": [c], "outputs": [text], "finish_reasons": [finish],
                              "output_tokens": [tok[1]]})
        return text

    def accept(candidates, guides):
        ok = [passes(extract_code(c), public, task["setup"]) for c in candidates]
        passing = [i for i, p in enumerate(ok) if p]
        chosen = rng.choice(passing) if passing else None
        rec["stages"][-1]["vote"] = {"passes_public": ok, "accepted": chosen is not None,
                                     "chosen": chosen}
        return (candidates[chosen], guides[chosen]) if passing else (None, None)

    # Stage 1: small model states n goals, then executes once per goal.
    goals = small("1a", [conv(config.system_prompt_guide_small, problem)] * config.n, 0.8, guide=True)
    cands = small("1b", [conv(config.system_prompt,
                              config.user_prompt_guide_1.format(problem=problem, guide_small=g))
                         for g in goals], 0.4)
    pred, rec["guide_small"] = accept(cands, goals)
    if pred is not None:
        return {**rec, "pred": pred, "round": 1}

    # Stage 2: large model writes a guideline; small model executes n times.
    # As in the released maths code, a failed stage 1 leaves guide_small as None.
    rec["guide_big"] = big("2a", conv(config.system_prompt_guide_big, problem), 0.8)
    prompt = config.user_prompt_guide_2.format(
        problem=problem, guide_small=rec["guide_small"], guide_big=rec["guide_big"])
    cands = small("2b", [conv(config.system_prompt, prompt)] * config.n, 0.8)
    pred, _ = accept(cands, cands)
    if pred is not None:
        return {**rec, "pred": pred, "round": 2}

    # Stage 3: large model executes with its own guideline (greedy, no check).
    pred = big("3", conv(
        config.system_prompt,
        config.user_prompt_guide_3.format(problem=problem, guide_big=rec["guide_big"])), 0)
    return {**rec, "pred": pred, "round": 3}


def direct_code(task, config, llm_big, llm_small, rng):
    """Single-model baseline: one greedy solution, no tests consulted."""
    rec = {"small_in": 0, "small_out": 0, "big_in": 0, "big_out": 0,
           "guide_small": None, "guide_big": None, "round": 0}
    c = conv(config.system_prompt, task["problem"])
    if config.direct_model == "big":
        pred, tok, _ = big_generate(llm_big, config, c, 0)
        rec["big_in"], rec["big_out"] = tok
    else:
        texts, tokens, _ = small_generate(llm_small, config, [c], 0)
        pred = texts[0]
        rec["small_in"], rec["small_out"] = tokens[0]
    return {**rec, "pred": pred}


def big_plan_code(task, config, llm_big, llm_small, rng):
    """Large model plans and executes: COPE's stage 3 for every task, no tests consulted."""
    rec = {"small_in": 0, "small_out": 0, "big_in": 0, "big_out": 0,
           "guide_small": None, "round": 0}
    rec["guide_big"], tok, _ = big_generate(
        llm_big, config, conv(config.system_prompt_guide_big, task["problem"]), 0.8)
    rec["big_in"] += tok[0]; rec["big_out"] += tok[1]
    pred, tok, _ = big_generate(llm_big, config, conv(
        config.system_prompt,
        config.user_prompt_guide_3.format(problem=task["problem"], guide_big=rec["guide_big"])), 0)
    rec["big_in"] += tok[0]; rec["big_out"] += tok[1]
    return {**rec, "pred": pred}


CODE_APPROACHES = {"collaborative": cope_code, "direct": direct_code, "big_plan": big_plan_code}
