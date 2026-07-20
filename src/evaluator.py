from c2_decision_utils import aggregate_decision, aggregate_vote
from c2_prompts import C2_OUTPUT_SUFFIX


def get_instruction_suffix(args):
    if args.data != "c2_decision":
        raise ValueError(f"Unsupported dataset: {args.data}")
    return C2_OUTPUT_SUFFIX


def evaluate_c2(
    responses,
    answer,
    sample,
    solver="decision",
    epsilon=0.25,
    safety_veto=True,
    debate_mode=False,
):
    if solver == "decision":
        final_answers, aggregate_answer, diagnostics = aggregate_decision(
            responses,
            sample,
            epsilon=epsilon,
            safety_veto=safety_veto,
            debate_mode=debate_mode,
        )
    elif solver == "vote":
        final_answers, aggregate_answer, diagnostics = aggregate_vote(responses)
    else:
        raise ValueError(f"Unsupported solver: {solver}")
    return final_answers, aggregate_answer, aggregate_answer == answer, diagnostics
