# cs470: single-model baselines for COPE ("DeepSeek only" and "local model only"
# in the CS470 proposal). Each model solves the problem once, at temperature 0,
# with COPE's own solution prompt, and records tokens the same way COPE does.

from sal.config import Config
from sal.search.collaborative import finalize_outputs, generate_big, generate_small


def direct(x, config: Config, llm_big, llm_small):
    items = [{"problem": problem, "pred": None, "guide_big": None,
              "guides_small": [""] * config.n, "guide_small": None,
              "small_in": 0, "small_out": 0, "big_in": 0, "big_out": 0, "round": 0}
             for problem in x["problem"]]
    idxs = list(range(len(items)))
    convs = [[{"role": "system", "content": config.system_prompt},
              {"role": "user", "content": item["problem"]}] for item in items]

    if config.direct_model == "big":
        items = generate_big(convs, config, llm_big, idxs, items, temperature=0, target="pred")
    else:
        # One completion per problem (recipe sets n: 1); threshold 0 accepts it.
        items = generate_small(convs, config, llm_small, idxs, items, temperature=0,
                               target="pred", duplicate=False, thresh=0)
    return finalize_outputs(x, items)


def big_plan(x, config: Config, llm_big, llm_small):
    """Large model plans and executes: COPE's stage 3 for every problem.

    The large model writes COPE's guideline (temperature 0.8), then solves the
    problem with it (temperature 0), with the same prompts COPE uses in stage 3.
    """
    items = [{"problem": problem, "pred": None, "guide_big": None,
              "guides_small": [""] * config.n, "guide_small": None,
              "small_in": 0, "small_out": 0, "big_in": 0, "big_out": 0, "round": 0}
             for problem in x["problem"]]
    idxs = list(range(len(items)))
    guide_convs = [[{"role": "system", "content": config.system_prompt_guide_big},
                    {"role": "user", "content": item["problem"]}] for item in items]
    items = generate_big(guide_convs, config, llm_big, idxs, items, temperature=0.8, target="guide")
    solve_convs = [[{"role": "system", "content": config.system_prompt},
                    {"role": "user", "content": config.user_prompt_guide_3.format(
                        problem=item["problem"], guide_big=item["guide_big"])}] for item in items]
    items = generate_big(solve_convs, config, llm_big, idxs, items, temperature=0, target="pred")
    return finalize_outputs(x, items)
