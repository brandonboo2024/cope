# cs470: re-score saved MBPP outputs (no model calls), e.g. after a grader fix.
#     python scripts/regrade_mbpp.py outputs/cs470/mbpp_big.jsonl
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from run_mbpp import load_tasks, summarise  # noqa: E402
from sal.search.code import extract_code, passes  # noqa: E402

for path in map(Path, sys.argv[1:]):
    rows = [json.loads(line) for line in path.open()]
    tasks = {t["task_id"]: t for t in load_tasks(len(rows))[0]}
    for r in rows:
        t, code = tasks[r["task_id"]], extract_code(r["pred"])
        r["passes_public"] = passes(code, t["public_tests"], t["setup"])
        r["correct"] = passes(code, t["hidden_tests"], t["setup"], timeout=30)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    summarise(rows, path, excluded=[])
