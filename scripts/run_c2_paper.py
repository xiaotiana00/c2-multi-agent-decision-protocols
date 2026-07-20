import argparse
import glob
import json
import shlex
import subprocess
import sys
from pathlib import Path


CHECKPOINTS = {
    "qwen2.5-7b": "Qwen/Qwen2.5-7B-Instruct",
    "ollama": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
}

GENERATION_ARGS = [
    "--c2_temperature", "0.14",
    "--c2_top_p", "0.78",
    "--c2_top_k", "8",
    "--c2_max_new_tokens", "64",
]

METHODS = {
    "single_vote": ["--num_agents", "1", "--debate_rounds", "0", "--solver", "vote", *GENERATION_ARGS],
    "multi_vote": ["--num_agents", "3", "--debate_rounds", "0", "--solver", "vote", *GENERATION_ARGS],
    "multi_debate_vote_t1": ["--num_agents", "3", "--debate_rounds", "1", "--solver", "vote", *GENERATION_ARGS],
    "multi_debate_vote_t2": ["--num_agents", "3", "--debate_rounds", "2", "--solver", "vote", *GENERATION_ARGS],
    "multi_debate_vote_t3": ["--num_agents", "3", "--debate_rounds", "3", "--solver", "vote", *GENERATION_ARGS],
    "multi_decision": ["--num_agents", "3", "--debate_rounds", "0", "--solver", "decision", "--multi_persona", *GENERATION_ARGS],
    "multi_decision_debate_t1": ["--num_agents", "3", "--debate_rounds", "1", "--solver", "decision", "--multi_persona", *GENERATION_ARGS],
    "multi_decision_debate_t2": ["--num_agents", "3", "--debate_rounds", "2", "--solver", "decision", "--multi_persona", *GENERATION_ARGS],
    "multi_decision_debate_t3": ["--num_agents", "3", "--debate_rounds", "3", "--solver", "decision", "--multi_persona", *GENERATION_ARGS],
    "mdd_t3_no_veto": ["--num_agents", "3", "--debate_rounds", "3", "--solver", "decision", "--multi_persona", "--c2_disable_safety_veto", *GENERATION_ARGS],
    "mdd_t3_no_role": ["--num_agents", "3", "--debate_rounds", "3", "--solver", "decision", *GENERATION_ARGS],
}

MAIN_METHODS = (
    "single_vote",
    "multi_vote",
    "multi_debate_vote_t1",
    "multi_debate_vote_t2",
    "multi_debate_vote_t3",
    "multi_decision",
    "multi_decision_debate_t1",
    "multi_decision_debate_t2",
    "multi_decision_debate_t3",
)

ABLATION_METHODS = (
    "mdd_t3_no_veto",
    "mdd_t3_no_role",
)

QWEN_METHODS = MAIN_METHODS + ABLATION_METHODS
DEEPSEEK_METHODS = (
    "single_vote",
    "multi_vote",
    "multi_debate_vote_t3",
    "multi_decision",
    "multi_decision_debate_t3",
    *ABLATION_METHODS,
)


def _value(tokens, flag, default=None):
    if flag not in tokens:
        return default
    index = tokens.index(flag)
    return tokens[index + 1] if index + 1 < len(tokens) else default


def _history_path(out_dir, run_tag):
    matches = glob.glob(str(out_dir / "history" / f"*__{run_tag}.jsonl"))
    return sorted(matches)[-1] if matches else None


def _run_config(history_path):
    if not history_path:
        return {}
    with open(history_path, "r", encoding="utf-8") as handle:
        first_line = next((line for line in handle if line.strip()), "")
    return (json.loads(first_line).get("run_config") or {}) if first_line else {}


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--main_py", default="src/main.py")
    parser.add_argument("--out_dir", default="out_paper")
    parser.add_argument("--model", choices=sorted(CHECKPOINTS), default="qwen2.5-7b")
    parser.add_argument("--data_size", type=int, default=100)
    parser.add_argument("--c2_data_file", default="")
    parser.add_argument("--seed_list", default="42,43,44")
    parser.add_argument("--methods", default="")
    parser.add_argument("--extra_args", default="")
    parser.add_argument("--workdir", default=".")
    args = parser.parse_args()

    workdir = Path(args.workdir).resolve()
    out_dir = (workdir / args.out_dir).resolve()
    (out_dir / "history").mkdir(parents=True, exist_ok=True)
    seeds = [int(value.strip()) for value in args.seed_list.split(",") if value.strip()]
    default_methods = QWEN_METHODS if args.model == "qwen2.5-7b" else DEEPSEEK_METHODS
    methods = [value.strip() for value in args.methods.split(",") if value.strip()] or list(default_methods)
    extras = shlex.split(args.extra_args)

    manifest_path = out_dir / "paper_manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as manifest:
        for method in methods:
            if method not in METHODS:
                raise ValueError(f"Unknown method: {method}")
            for seed in seeds:
                run_tag = f"paper-{method}-seed{seed}"
                command = [
                    args.python,
                    args.main_py,
                    "--data", "c2_decision",
                    "--model", args.model,
                    "--seed", str(seed),
                    "--out_dir", str(out_dir),
                    "--data_size", str(args.data_size),
                    "--c2_version", "v4",
                    "--run_tag", run_tag,
                    *METHODS[method],
                    *extras,
                ]
                if args.c2_data_file:
                    command.extend(["--c2_data_file", args.c2_data_file])
                subprocess.run(command, cwd=workdir, check=True)

                history_path = _history_path(out_dir, run_tag)
                config = _run_config(history_path)
                solver = _value(command, "--solver")
                record = {
                    "method": method,
                    "protocol": method,
                    "seed": seed,
                    "data": "c2_decision",
                    "data_size": args.data_size,
                    "c2_version": "v4",
                    "c2_loaded_file": config.get("c2_loaded_file"),
                    "c2_dataset_sha256": config.get("c2_dataset_sha256"),
                    "c2_dataset_metadata_file": config.get("c2_dataset_metadata_file"),
                    "model": args.model,
                    "formal_checkpoint": config.get("formal_checkpoint") or CHECKPOINTS[args.model],
                    "inference_backend": config.get("inference_backend"),
                    "runtime_environment": config.get("runtime_environment"),
                    "ollama_model": config.get("ollama_model"),
                    "ollama_quantization": config.get("ollama_quantization"),
                    "ollama_model_digest": config.get("ollama_model_digest"),
                    "ollama_version": config.get("ollama_version"),
                    "ollama_architecture": config.get("ollama_architecture"),
                    "ollama_parameter_count": config.get("ollama_parameter_count"),
                    "ollama_context_length": config.get("ollama_context_length"),
                    "ollama_embedding_length": config.get("ollama_embedding_length"),
                    "do_sample": True,
                    "temperature": float(_value(command, "--c2_temperature")),
                    "top_p": float(_value(command, "--c2_top_p")),
                    "top_k": int(_value(command, "--c2_top_k")),
                    "max_new_tokens": int(_value(command, "--c2_max_new_tokens")),
                    "num_agents": int(_value(command, "--num_agents")),
                    "debate_rounds": int(_value(command, "--debate_rounds")),
                    "solver": solver,
                    "multi_persona": "--multi_persona" in command,
                    "profile_source": "heuristic" if solver == "decision" else "not_applicable",
                    "c2_epsilon": config.get("c2_epsilon"),
                    "veto_setting": "not_applicable" if solver == "vote" else ("disabled" if "--c2_disable_safety_veto" in command else "enabled"),
                    "prompt_hash": config.get("prompt_hash"),
                    "run_tag": run_tag,
                    "cmd": command,
                    "history_path": history_path,
                }
                manifest.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Saved manifest: {manifest_path}")


if __name__ == "__main__":
    main()
