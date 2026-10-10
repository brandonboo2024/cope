# cs470: write an org file that traces every saved input and output, task by task.
#
#     python scripts/trace_cs470.py math500     # -> outputs/cs470/trace_math500.org
#     python scripts/trace_cs470.py mbpp        # -> outputs/cs470/trace_mbpp.org
#
# For each task: the problem and reference, then for each system (Qwen alone,
# DeepSeek alone, COPE, DeepSeek plan -> DeepSeek) each stage's prompt, rebuilt
# from the recipe's templates exactly as the code sends it, and the saved
# output. The runs did not save the candidates a stage rejected (COPE's 8
# stage-1 and stage-2 solutions, their answers or test results), so those are
# marked as not saved; only the accepted candidate survives, as the final pred.

import json
import sys
from pathlib import Path

import yaml

from sal.config import Config

OUT = Path("outputs/cs470")
SYSTEMS = [("small", "Qwen2.5-1.5B alone"), ("big", "DeepSeek alone"),
           ("cope", "COPE"), ("big_plan", "DeepSeek plan -> DeepSeek")]
NOT_SAVED = "/not saved by this run/"


def prompts(bench):
    """The recipe's prompts, falling back to Config's defaults (the maths prompts)."""
    recipe = yaml.safe_load(Path(f"recipes/cs470/{bench}_cope.yaml").read_text())
    keys = ["system_prompt", "system_prompt_guide_big", "system_prompt_guide_small",
            "user_prompt_guide_1", "user_prompt_guide_2", "user_prompt_guide_3"]
    return {k: recipe.get(k, getattr(Config, k)) for k in keys}


def block(text):
    lines = [("," + l if l.startswith(("*", "#+")) else l) for l in str(text).splitlines()]
    return "#+begin_example\n" + "\n".join(lines) + "\n#+end_example\n"


def call(title, system, user, output, note=""):
    shown = output if output.startswith(NOT_SAVED) else block(output)
    prompt = block("[system]\n" + system + "\n\n[user]\n" + user)
    return f"*** {title}\n{note}**** Input\n{prompt}**** Output\n{shown}\n"


def load(bench):
    rows = {}
    for system, _ in SYSTEMS:
        path = OUT / f"{bench}_{system}.jsonl"
        recs = [json.loads(line) for line in path.open()]
        if bench == "math500":  # correctness lives in the graded summary
            graded = json.loads(path.with_suffix(".graded.json").read_text())["tasks_detail"]
            for rec, g in zip(recs, graded):
                assert rec["unique_id"] == g["unique_id"]
                rec.update(correct=g["correct"], extracted=g["predicted"], key=g["unique_id"])
        else:
            for rec in recs:
                rec["key"] = rec["task_id"]
        rows[system] = {r["key"]: r for r in recs}
    return rows


def mbpp_problems():
    from datasets import load_dataset
    out = {}
    for r in load_dataset("google-research-datasets/mbpp", "full", split="test"):
        out[r["task_id"]] = (r["text"] + "\nYour code should pass these tests:\n"
                             + "\n".join(r["test_list"]))
    return out


def stages(rec, problem, p, bench):
    """COPE's stages for one task, as far as the saved record allows."""
    rnd = rec["round"]
    goals = rec.get("guides_small") or []
    s = []
    goal_out = ("\n".join(f"{i}. {g}" for i, g in enumerate(goals, 1)) if any(goals)
                else NOT_SAVED + (f" (accepted goal: {rec['guide_small']!r})"
                                  if rnd == 1 else ""))
    s.append(call("Stage 1a: Qwen states the goal (8 samples, temperature 0.8)",
                  p["system_prompt_guide_small"], problem + "\n[assistant prefix] The goal",
                  goal_out))
    accept = ("the most common answer appears in more than 75% of the 8 solutions"
              if bench == "math500" else "at least one solution passes the public tests")
    first = rec["pred"] if rnd == 1 else NOT_SAVED
    s.append(call("Stage 1b: Qwen solves once per goal (8 solutions, temperature 0.4)",
                  p["system_prompt"],
                  p["user_prompt_guide_1"].format(problem=problem,
                                                  guide_small=rec["guide_small"] or "<goal i>"),
                  first,
                  f"Accepted if {accept}. Stage 1 {'ACCEPTED' if rnd == 1 else 'REJECTED'}; "
                  f"only the accepted solution is saved.\n"))
    if rnd == 1:
        return s
    s.append(call("Stage 2a: DeepSeek writes a guideline (temperature 0.8)",
                  p["system_prompt_guide_big"], problem, rec["guide_big"]))
    accept2 = ("the most common answer appears in more than 50% of the 8 solutions"
               if bench == "math500" else "at least one solution passes the public tests")
    s.append(call("Stage 2b: Qwen solves with the guideline (8 solutions, temperature 0.8)",
                  p["system_prompt"],
                  p["user_prompt_guide_2"].format(problem=problem, guide_small=None,
                                                  guide_big=rec["guide_big"]),
                  rec["pred"] if rnd == 2 else NOT_SAVED,
                  f"Goal is None because stage 1 failed (the released code's behaviour). "
                  f"Accepted if {accept2}. Stage 2 {'ACCEPTED' if rnd == 2 else 'REJECTED'}.\n"))
    if rnd == 2:
        return s
    s.append(call("Stage 3: DeepSeek solves with its guideline (temperature 0)",
                  p["system_prompt"],
                  p["user_prompt_guide_3"].format(problem=problem, guide_big=rec["guide_big"]),
                  rec["pred"]))
    return s


def verdict(rec, bench):
    ok = "CORRECT" if rec["correct"] else "WRONG"
    if bench == "math500":
        return f"{ok}, extracted answer {rec['extracted']!r}"
    return f"{ok} (hidden MBPP-ET tests), passes public tests: {rec['passes_public']}"


def main():
    bench = sys.argv[1]
    p = prompts(bench)
    rows = load(bench)
    keys = list(rows["cope"])
    problems = mbpp_problems() if bench == "mbpp" else {k: r["problem"] for k, r in rows["cope"].items()}
    head = [f"#+title: Trace of every saved input and output, {bench} (COPE fork, branch cs470)",
            "#+startup: overview", "",
            "Prompts are rebuilt from the recipe's templates as the code sends them. COPE's rejected",
            "candidates were not saved, so stages that rejected show only their input.", "",
            "* Summary (1 = correct; COPE shows the stage that produced its answer)",
            "| # | task | " + " | ".join(n for _, n in SYSTEMS) + " |", "|-"]
    body = []
    for n, key in enumerate(keys, 1):
        cells = []
        for system, _ in SYSTEMS:
            r = rows[system][key]
            cells.append(f"{int(r['correct'])}" + (f" (stage {r['round']})" if system == "cope" else ""))
        head.append(f"| {n} | [[*T{n}][{key}]] | " + " | ".join(cells) + " |")
        cope = rows["cope"][key]
        ref = cope.get("answer") if bench == "math500" else "hidden MBPP-ET tests"
        body.append(f"* T{n}\n*Task:* {key}\n\n*Problem:*\n{block(problems[key])}\n*Reference:* {ref}")
        for system, name in SYSTEMS:
            r = rows[system][key]
            body.append(f"** {name}: {verdict(r, bench)}")
            if system in ("small", "big"):
                body.append(call("Solve directly (temperature 0)", p["system_prompt"],
                                 problems[key], r["pred"]))
            elif system == "big_plan":
                body.append(call("DeepSeek writes a guideline (temperature 0.8)",
                                 p["system_prompt_guide_big"], problems[key], r["guide_big"]))
                body.append(call("DeepSeek solves with it (temperature 0)", p["system_prompt"],
                                 p["user_prompt_guide_3"].format(problem=problems[key],
                                                                 guide_big=r["guide_big"]),
                                 r["pred"]))
            else:
                body += stages(r, problems[key], p, bench)
    path = OUT / f"trace_{bench}.org"
    path.write_text("\n".join(head + body) + "\n")
    print(f"{len(keys)} tasks -> {path}")


if __name__ == "__main__":
    main()
