# cs470: grade COPE's saved MATH-500 outputs against the reference answers.
#
# Uses the Qwen2.5-Math grader bundled with COPE (sal/utils/qwen_math_parser.py):
# the final answer is extracted from the solution (boxed or "final answer is"),
# then compared numerically or symbolically with the reference.
#
#     python scripts/grade_math.py outputs/cs470/math500_cope.jsonl ...
# Writes <file>.graded.json next to each input.

import json
import sys
from pathlib import Path

from sal.utils.qwen_math_parser import extract_answer, math_equal, strip_string

PRICE_IN, PRICE_OUT = 0.30, 1.20  # DeepSeek Flash peak US$ per million tokens


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
        "tasks_detail": graded,
    }
    out = Path(path).with_suffix(".graded.json")
    out.write_text(json.dumps(summary, indent=1))
    print(f"{path}: {summary['correct']}/{n} correct, "
          f"{summary['tokens_per_task']:.0f} tokens/task, ${summary['usd_total']:.4f}")


if __name__ == "__main__":
    for p in sys.argv[1:]:
        grade(p)
