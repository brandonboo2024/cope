# cs470: grade COPE's saved MATH-500 outputs against the reference answers.
#
# Uses the Qwen2.5-Math grader bundled with COPE (sal/utils/qwen_math_parser.py):
# the final answer is extracted from the solution (boxed or "final answer is"),
# then compared numerically or symbolically with the reference.
#
#     python scripts/grade_math.py outputs/cs470/math500_cope.jsonl ...
# Writes <file>.graded.json next to each input.
#
# cs470: when the run logged its stages (a "stages" field), every attempt of
# every solving stage is graded too, so a rejected stage shows how many of its
# attempts were right, and the summary counts wrong accepts and wrong rejects.

import json
import sys
from pathlib import Path

from sal.utils.qwen_math_parser import extract_answer, math_equal, strip_string

PRICE_IN, PRICE_OUT = 0.30, 1.20  # DeepSeek Flash peak US$ per million tokens


SOLVING = {"1b", "2b", "3", "direct", "solve"}


def is_correct(predicted, reference):
    return bool(math_equal(predicted, reference, timeout=True))


def grade_stages(r, reference):
    """Each solving stage's attempts, graded, with the stage's vote and verdict."""
    out = []
    for st in json.loads(r.get("stages") or "[]"):
        if st["stage"] not in SOLVING:
            continue
        answers = [extract_answer(o, "math") for o in st["outputs"]]
        right = [is_correct(a, reference) for a in answers]
        vote = st.get("vote") or {}
        out.append({
            "stage": st["stage"], "attempts": len(right), "right": sum(right),
            "answers": answers, "correct": right,
            "finish_reasons": st["finish_reasons"], "output_tokens": st["output_tokens"],
            "majority": vote.get("majority"), "share": vote.get("share"),
            "threshold": vote.get("threshold"), "accepted": vote.get("accepted"),
            "majority_correct": (is_correct(extract_answer(st["outputs"][vote["answers"].index(vote["majority"])], "math"), reference)
                                 if vote else None),
        })
    return out


def grade(path):
    rows = [json.loads(line) for line in Path(path).open() if line.strip()]
    graded = []
    for r in rows:
        predicted = extract_answer(r["pred"], "math")
        reference = strip_string(r["answer"])
        graded.append({
            "unique_id": r.get("unique_id"), "level": r.get("level"),
            "reference": reference, "predicted": predicted,
            "correct": bool(math_equal(predicted, reference, timeout=True)),
            "round": r["round"],  # COPE stage that produced the answer (0 = direct)
            "small_in": r["small_in"], "small_out": r["small_out"],
            "big_in": r["big_in"], "big_out": r["big_out"],
            "usd": (r["big_in"] * PRICE_IN + r["big_out"] * PRICE_OUT) / 1e6,
            "stages": grade_stages(r, reference),
        })
    n = len(graded)
    total = lambda k: sum(g[k] for g in graded)
    summary = {
        "file": str(path), "tasks": n, "correct": total("correct"),
        "accuracy": total("correct") / n,
        "tokens_per_task": (total("small_in") + total("small_out")
                            + total("big_in") + total("big_out")) / n,
        "small_tokens_per_task": (total("small_in") + total("small_out")) / n,
        "big_tokens_per_task": (total("big_in") + total("big_out")) / n,
        "usd_total": total("usd"),
        "rounds": {str(k): sum(g["round"] == k for g in graded) for k in (0, 1, 2, 3)},
        "stage_checks": stage_checks(graded),
        "tasks_detail": graded,
    }
    out = Path(path).with_suffix(".graded.json")
    if summary["stage_checks"]:
        print(json.dumps(summary["stage_checks"]))
    out.write_text(json.dumps(summary, indent=1))
    print(f"{path}: {summary['correct']}/{n} correct, "
          f"{summary['tokens_per_task']:.0f} tokens/task, ${summary['usd_total']:.4f}")


def stage_checks(graded):
    """Per voting stage: how often it accepted a wrong majority or rejected a right one."""
    checks = {}
    for g in graded:
        for st in g["stages"]:
            if st["accepted"] is None:
                continue
            c = checks.setdefault(st["stage"], {"ran": 0, "accepted": 0, "accepted_wrong": 0,
                                                "rejected_right_majority": 0,
                                                "rejected_any_right": 0, "capped_attempts": 0})
            c["ran"] += 1
            c["accepted"] += st["accepted"]
            c["accepted_wrong"] += st["accepted"] and not st["majority_correct"]
            c["rejected_right_majority"] += (not st["accepted"]) and st["majority_correct"]
            c["rejected_any_right"] += (not st["accepted"]) and st["right"] > 0
            c["capped_attempts"] += sum(f == "length" for f in st["finish_reasons"])
    return checks


if __name__ == "__main__":
    for p in sys.argv[1:]:
        grade(p)
