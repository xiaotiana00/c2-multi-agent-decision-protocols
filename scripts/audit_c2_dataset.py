import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.c2_decision_utils import METRIC_FIELDS, OPTIONS, utility_from_profile


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def dominates_label(label, profiles):
    others = [opt for opt in OPTIONS if opt != label]
    mission_ok = all(profiles["mission_gain"][label] >= profiles["mission_gain"][o] for o in others)
    risk_ok = all(
        all(profiles[m][label] <= profiles[m][o] for o in others)
        for m in METRIC_FIELDS if m != "mission_gain"
    )
    return mission_ok and risk_ok


def dataset_profiles(sample):
    profiles = {}
    for metric in METRIC_FIELDS:
        values = sample.get(metric)
        if not isinstance(values, dict) or any(option not in values for option in OPTIONS):
            raise ValueError(f"Sample {sample.get('id')} has no complete {metric} annotation.")
        profiles[metric] = {option: float(values[option]) for option in OPTIONS}
    return profiles


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--data_file", type=str, required=True)
    parser.add_argument("--epsilon", type=float, default=0.25)
    parser.add_argument("--out_json", type=str, default="")
    args = parser.parse_args()

    rows = load_jsonl(args.data_file)
    audit_rows = []
    easy_count = 0
    only_one_feasible = 0
    label_top_utility = 0

    for row in rows:
        profiles = dataset_profiles(row)
        label = str(row["label"]).strip().upper()
        utilities = {opt: utility_from_profile("None", profiles, opt) for opt in OPTIONS}
        ranked = sorted(OPTIONS, key=lambda opt: utilities[opt], reverse=True)
        feasible = [opt for opt in OPTIONS if profiles["catastrophic_risk"][opt] <= args.epsilon]
        margin = utilities[ranked[0]] - utilities[ranked[1]]
        dominates = dominates_label(label, profiles)
        is_easy = (ranked[0] == label and (len(feasible) == 1 or (dominates and margin >= 0.20)))
        easy_count += int(is_easy)
        only_one_feasible += int(len(feasible) == 1)
        label_top_utility += int(ranked[0] == label)
        audit_rows.append({
            "id": row.get("id"),
            "label": label,
            "profile_source": "evaluation_annotation",
            "ranked_by_utility": ranked,
            "utility_margin_top2": round(margin, 4),
            "feasible_under_epsilon": feasible,
            "label_is_top_utility": ranked[0] == label,
            "label_dominates_all_metrics": dominates,
            "flag_easy": is_easy,
        })

    summary = {
        "n": len(rows),
        "epsilon": args.epsilon,
        "label_top_utility_rate": label_top_utility / max(len(rows), 1),
        "only_one_feasible_rate": only_one_feasible / max(len(rows), 1),
        "easy_case_rate": easy_count / max(len(rows), 1),
        "flagged_samples": [r for r in audit_rows if r["flag_easy"]],
    }

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.out_json:
        Path(args.out_json).write_text(json.dumps({"summary": summary, "rows": audit_rows}, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
