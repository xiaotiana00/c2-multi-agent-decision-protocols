import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
import random
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def history_metrics(history_path: str):
    rows = load_jsonl(history_path)
    if not rows:
        return None
    final_correct = []
    final_risk = []
    final_cat = []
    final_dis = []
    final_change = []
    changed_agent_pairs = 0
    valid_agent_pairs = 0
    paired_change_available = True
    has_debate_round = False
    round0_correct = []
    total_final_outputs = 0
    unrecoverable_final_outputs = 0
    schema_invalid_final_outputs = 0
    sample_rows = []
    for row in rows:
        rounds = row["rounds"]
        r0 = rounds["0"]
        round_keys = sorted(rounds, key=int)
        rk = round_keys[-1]
        rf = rounds[rk]
        round0_correct.append(int(bool(r0.get("aggregate_answer_iscorr"))))
        final_correct.append(int(bool(rf.get("aggregate_answer_iscorr"))))
        final_risk.append(int(bool(rf.get("risk_hit"))))
        final_cat.append(int(bool(rf.get("catastrophic_hit"))))
        final_dis.append(float(rf.get("disagreement_rate", 0.0)))
        final_change.append(float(rf.get("answer_change_rate", 0.0)))
        if len(round_keys) > 1:
            has_debate_round = True
            previous = rounds[round_keys[-2]].get("final_answers")
            current = rf.get("final_answers")
            if isinstance(previous, list) and isinstance(current, list) and len(previous) == len(current):
                pairs = [(a, b) for a, b in zip(previous, current) if a and b]
                changed_agent_pairs += sum(a != b for a, b in pairs)
                valid_agent_pairs += len(pairs)
            else:
                paired_change_available = False
        parsed_responses = rf.get("parsed_responses") or {}
        auditable_responses = [
            parsed for parsed in parsed_responses.values()
            if "recoverable" in parsed and "schema_valid" in parsed
        ]
        total_final_outputs += len(auditable_responses)
        unrecoverable_final_outputs += sum(
            int(not bool(parsed["recoverable"])) for parsed in auditable_responses
        )
        schema_invalid_final_outputs += sum(
            int(not bool(parsed["schema_valid"])) for parsed in auditable_responses
        )
        sample_rows.append({
            "sample_id": row.get("sample_id"),
            "r0_correct": int(bool(r0.get("aggregate_answer_iscorr"))),
            "final_correct": int(bool(rf.get("aggregate_answer_iscorr"))),
            "final_risk": int(bool(rf.get("risk_hit"))),
            "final_cat": int(bool(rf.get("catastrophic_hit"))),
        })
    return {
        "n": len(rows),
        "round0_acc": mean(round0_correct),
        "final_acc": mean(final_correct),
        "acc_gain": mean(final_correct) - mean(round0_correct),
        "final_risk_rate": mean(final_risk),
        "final_cat_rate": mean(final_cat),
        "final_disagreement": mean(final_dis),
        "final_change_rate": (
            changed_agent_pairs / valid_agent_pairs if valid_agent_pairs else 0.0
        ) if has_debate_round and paired_change_available else mean(final_change),
        "total_final_agent_outputs": total_final_outputs,
        "unrecoverable_final_outputs": unrecoverable_final_outputs,
        "schema_invalid_final_outputs": schema_invalid_final_outputs,
        "sample_rows": sample_rows,
    }


def summarize_group(values):
    if not values:
        return {"mean": None, "std": None}
    if len(values) == 1:
        return {"mean": values[0], "std": 0.0}
    return {"mean": mean(values), "std": stdev(values)}


def paired_bootstrap_diff(a_rows, b_rows, key="final_correct", iters=4000, seed=1234):
    a_map = {r["sample_id"]: r[key] for r in a_rows}
    b_map = {r["sample_id"]: r[key] for r in b_rows}
    common = sorted(set(a_map) & set(b_map))
    if not common:
        return None
    diffs = [a_map[i] - b_map[i] for i in common]
    obs = sum(diffs) / len(diffs)
    rng = random.Random(seed)
    boots = []
    for _ in range(iters):
        idxs = [rng.randrange(len(diffs)) for _ in range(len(diffs))]
        boots.append(sum(diffs[i] for i in idxs) / len(idxs))
    boots.sort()
    lo = boots[int(0.025 * len(boots))]
    hi = boots[int(0.975 * len(boots))]
    p = 2 * min(
        sum(1 for x in boots if x >= 0) / len(boots),
        sum(1 for x in boots if x <= 0) / len(boots),
    )
    return {"diff": obs, "ci95": [lo, hi], "p_boot": min(1.0, p)}


def _paired_stat_for_key(per_method_seed_rows, ma, mb, key):
    common_seeds = sorted(set(per_method_seed_rows[ma]) & set(per_method_seed_rows[mb]))
    per_seed_diffs = []
    for seed in common_seeds:
        a_map = {r["sample_id"]: r[key] for r in per_method_seed_rows[ma][seed]}
        b_map = {r["sample_id"]: r[key] for r in per_method_seed_rows[mb][seed]}
        common_ids = sorted(set(a_map) & set(b_map))
        if common_ids:
            per_seed_diffs.append([a_map[sample_id] - b_map[sample_id] for sample_id in common_ids])
    if not per_seed_diffs:
        return None

    observed = mean(mean(seed_diffs) for seed_diffs in per_seed_diffs)
    rng = random.Random(1234)
    boots = []
    for _ in range(4000):
        seed_means = []
        for seed_diffs in per_seed_diffs:
            sampled = [seed_diffs[rng.randrange(len(seed_diffs))] for _ in range(len(seed_diffs))]
            seed_means.append(mean(sampled))
        boots.append(mean(seed_means))
    boots.sort()
    lo = boots[int(0.025 * len(boots))]
    hi = boots[int(0.975 * len(boots))]
    p_boot = 2 * min(
        sum(1 for value in boots if value >= 0) / len(boots),
        sum(1 for value in boots if value <= 0) / len(boots),
    )
    return {
        "diff_mean": observed,
        "ci95_low_mean": lo,
        "ci95_high_mean": hi,
        "p_boot_mean": min(1.0, p_boot),
    }


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="")
    args = parser.parse_args()

    manifest_rows = load_jsonl(args.manifest)
    per_method = defaultdict(list)
    per_method_seed_rows = defaultdict(dict)

    for row in manifest_rows:
        history_path = row.get("history_path")
        if not history_path:
            continue
        metrics = history_metrics(history_path)
        if metrics is None:
            continue
        merged = {**row, **metrics}
        per_method[row["method"]].append(merged)
        per_method_seed_rows[row["method"]][row["seed"]] = metrics["sample_rows"]

    summary_rows = []
    for method, rows in sorted(per_method.items()):
        summary_rows.append({
            "method": method,
            "runs": len(rows),
            "n_per_run": rows[0]["n"] if rows else 0,
            "round0_acc_mean": summarize_group([r["round0_acc"] for r in rows])["mean"],
            "round0_acc_std": summarize_group([r["round0_acc"] for r in rows])["std"],
            "final_acc_mean": summarize_group([r["final_acc"] for r in rows])["mean"],
            "final_acc_std": summarize_group([r["final_acc"] for r in rows])["std"],
            "acc_gain_mean": summarize_group([r["acc_gain"] for r in rows])["mean"],
            "acc_gain_std": summarize_group([r["acc_gain"] for r in rows])["std"],
            "risk_rate_mean": summarize_group([r["final_risk_rate"] for r in rows])["mean"],
            "risk_rate_std": summarize_group([r["final_risk_rate"] for r in rows])["std"],
            "cat_rate_mean": summarize_group([r["final_cat_rate"] for r in rows])["mean"],
            "cat_rate_std": summarize_group([r["final_cat_rate"] for r in rows])["std"],
            "disagreement_mean": summarize_group([r["final_disagreement"] for r in rows])["mean"],
            "disagreement_std": summarize_group([r["final_disagreement"] for r in rows])["std"],
            "change_rate_mean": summarize_group([r["final_change_rate"] for r in rows])["mean"],
            "change_rate_std": summarize_group([r["final_change_rate"] for r in rows])["std"],
            "unrecoverable_parse_failures": sum(r["unrecoverable_final_outputs"] for r in rows),
            "schema_invalid_outputs": sum(r["schema_invalid_final_outputs"] for r in rows),
            "parse_audited_final_outputs": sum(r["total_final_agent_outputs"] for r in rows),
        })

    out_dir = Path(args.out_dir) if args.out_dir else Path(args.manifest).resolve().parent
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "paper_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()) if summary_rows else ["method"])
        writer.writeheader()
        writer.writerows(summary_rows)

    pairwise = []
    methods = sorted(per_method_seed_rows.keys())
    for i in range(len(methods)):
        for j in range(i + 1, len(methods)):
            ma, mb = methods[i], methods[j]
            common_seeds = sorted(set(per_method_seed_rows[ma]) & set(per_method_seed_rows[mb]))
            if not common_seeds:
                continue
            acc_stat = _paired_stat_for_key(per_method_seed_rows, ma, mb, "final_correct")
            risk_stat = _paired_stat_for_key(per_method_seed_rows, ma, mb, "final_risk")
            cat_stat = _paired_stat_for_key(per_method_seed_rows, ma, mb, "final_cat")
            if acc_stat:
                pairwise.append({
                    "method_a": ma,
                    "method_b": mb,
                    "common_seeds": common_seeds,
                    "acc_diff_mean": acc_stat["diff_mean"],
                    "acc_ci95_low_mean": acc_stat["ci95_low_mean"],
                    "acc_ci95_high_mean": acc_stat["ci95_high_mean"],
                    "acc_p_boot_mean": acc_stat["p_boot_mean"],
                    "risk_diff_mean": risk_stat["diff_mean"] if risk_stat else None,
                    "risk_ci95_low_mean": risk_stat["ci95_low_mean"] if risk_stat else None,
                    "risk_ci95_high_mean": risk_stat["ci95_high_mean"] if risk_stat else None,
                    "risk_p_boot_mean": risk_stat["p_boot_mean"] if risk_stat else None,
                    "cat_diff_mean": cat_stat["diff_mean"] if cat_stat else None,
                    "cat_ci95_low_mean": cat_stat["ci95_low_mean"] if cat_stat else None,
                    "cat_ci95_high_mean": cat_stat["ci95_high_mean"] if cat_stat else None,
                    "cat_p_boot_mean": cat_stat["p_boot_mean"] if cat_stat else None,
                })

    pairwise_path = out_dir / "paper_pairwise.json"
    pairwise_path.write_text(json.dumps(pairwise, ensure_ascii=False, indent=2), encoding="utf-8")

    print("Saved:")
    print(f"  {csv_path}")
    print(f"  {pairwise_path}")
    print("\nSummary:")
    for row in summary_rows:
        print(
            f"- {row['method']}: final_acc={row['final_acc_mean']:.4f}±{row['final_acc_std']:.4f}, "
            f"risk={row['risk_rate_mean']:.4f}, cat={row['cat_rate_mean']:.4f}, "
            f"disagreement={row['disagreement_mean']:.4f}"
        )


if __name__ == "__main__":
    main()
