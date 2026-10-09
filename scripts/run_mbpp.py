# cs470: run COPE (reimplemented for code) or a single-model baseline on MBPP,
# then score the final programs on the hidden MBPP-ET tests.
#
#     python scripts/run_mbpp.py recipes/cs470/mbpp_cope.yaml
#
# Task selection: MBPP test split in task_id order; a task is eligible only if
# MBPP's own reference solution passes both its public tests and its MBPP-ET
# tests (checked before any model runs). The first `num_samples` eligible
# tasks are used.

import ast
import json
import logging
import os
import random
from pathlib import Path

from datasets import load_dataset
from openai import OpenAI

from sal.config import Config
from sal.search.code import CODE_APPROACHES, extract_code, passes
from sal.utils.parser import H4ArgumentParser

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
PRICE_IN, PRICE_OUT = 0.30, 1.20  # DeepSeek Flash peak US$ per million tokens


def load_tasks(limit):
    mbpp = load_dataset("google-research-datasets/mbpp", "full", split="test")
    et = {int(r["task_id"]): r["test_list"] if isinstance(r["test_list"], list)
          else ast.literal_eval(r["test_list"])
          for r in load_dataset("dz1/CodeScore-MBPP-ET", split="train")}
    tasks, excluded = [], []
    for r in mbpp:
        if len(tasks) == limit:
            break
        hidden = et.get(r["task_id"])
        task = {
            "task_id": r["task_id"], "setup": r["test_setup_code"],
            "public_tests": r["test_list"], "hidden_tests": hidden,
            # Standard MBPP prompt (Austin et al. 2021): description + public tests.
            "problem": r["text"] + "\nYour code should pass these tests:\n"
                       + "\n".join(r["test_list"]),
        }
        ok = hidden is not None and passes(r["code"], r["test_list"], r["test_setup_code"]) \
            and passes(r["code"], hidden, r["test_setup_code"], timeout=30)
        (tasks if ok else excluded).append(task if ok else r["task_id"])
    return tasks, excluded


def main():
    config = H4ArgumentParser(Config).parse()
    approach = CODE_APPROACHES[config.approach]
    tasks, excluded = load_tasks(config.num_samples)
    logger.info(f"{len(tasks)} eligible tasks; reference failed on {excluded}")

    llm_big = OpenAI(api_key=os.environ["OPENAI_API_KEY"],
                     base_url=os.environ.get("OPENAI_BASE_URL"))
    llm_small = None
    if not (config.approach == "big_plan"
            or (config.approach == "direct" and config.direct_model == "big")):
        from vllm import LLM
        llm_small = LLM(model=config.model_path_small,
                        gpu_memory_utilization=config.gpu_memory_utilization,
                        enable_prefix_caching=True, enforce_eager=config.enforce_eager, seed=config.seed)

    rng = random.Random(config.seed)
    out_path = Path(config.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, task in enumerate(tasks, 1):
        rec = approach(task, config, llm_big, llm_small, rng)
        code = extract_code(rec["pred"])
        rec.update(task_id=task["task_id"],
                   passes_public=passes(code, task["public_tests"], task["setup"]),
                   correct=passes(code, task["hidden_tests"], task["setup"], timeout=30),
                   usd=(rec["big_in"] * PRICE_IN + rec["big_out"] * PRICE_OUT) / 1e6)
        rows.append(rec)
        logger.info(f"{i}/{len(tasks)} task {task['task_id']}: round {rec['round']}, "
                    f"correct={rec['correct']}")

    with out_path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    summarise(rows, out_path, excluded)


def summarise(rows, out_path, excluded):
    n = len(rows)
    tot = lambda k: sum(r[k] for r in rows)
    summary = {
        "file": str(out_path), "tasks": n, "excluded_task_ids": excluded,
        "correct": tot("correct"), "accuracy": tot("correct") / n,
        "passes_public": tot("passes_public"),
        "tokens_per_task": (tot("small_in") + tot("small_out") + tot("big_in") + tot("big_out")) / n,
        "small_tokens_per_task": (tot("small_in") + tot("small_out")) / n,
        "big_tokens_per_task": (tot("big_in") + tot("big_out")) / n,
        "usd_total": tot("usd"),
        "rounds": {str(k): sum(r["round"] == k for r in rows) for k in (0, 1, 2, 3)},
    }
    Path(out_path).with_suffix(".graded.json").write_text(json.dumps(summary, indent=1))
    logger.info(json.dumps({k: v for k, v in summary.items() if k != "excluded_task_ids"}))

if __name__ == "__main__":
    main()
