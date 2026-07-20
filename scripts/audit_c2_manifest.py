import argparse
import json
from pathlib import Path


REQUIRED_COMMAND_FIELDS = {
    "--model": "model identifier",
    "--debate_rounds": "debate-round setting",
    "--solver": "solver setting",
}

REQUIRED_METADATA_FIELDS = [
    "model",
    "formal_checkpoint",
    "inference_backend",
    "protocol",
    "seed",
    "do_sample",
    "temperature",
    "top_p",
    "top_k",
    "max_new_tokens",
    "num_agents",
    "debate_rounds",
    "solver",
    "multi_persona",
    "profile_source",
    "c2_epsilon",
    "veto_setting",
    "c2_version",
    "c2_loaded_file",
    "c2_dataset_sha256",
    "c2_dataset_metadata_file",
    "prompt_hash",
    "runtime_environment",
    "history_path",
]

PAPER_GENERATION_CONFIG = {
    "do_sample": True,
    "temperature": 0.14,
    "top_p": 0.78,
    "top_k": 8,
    "max_new_tokens": 64,
}

PROTOCOL_CONTROLS = {
    "single_vote": (1, 0, "vote", False),
    "multi_vote": (3, 0, "vote", False),
    "multi_debate_vote_t1": (3, 1, "vote", False),
    "multi_debate_vote_t2": (3, 2, "vote", False),
    "multi_debate_vote_t3": (3, 3, "vote", False),
    "multi_decision": (3, 0, "decision", True),
    "multi_decision_debate_t1": (3, 1, "decision", True),
    "multi_decision_debate_t2": (3, 2, "decision", True),
    "multi_decision_debate_t3": (3, 3, "decision", True),
    "mdd_t3_no_veto": (3, 3, "decision", True),
    "mdd_t3_no_role": (3, 3, "decision", False),
}

MODEL_PROTOCOLS = {
    "qwen2.5-7b": {
        "single_vote", "multi_vote", "multi_debate_vote_t1", "multi_debate_vote_t2",
        "multi_debate_vote_t3", "multi_decision", "multi_decision_debate_t1",
        "multi_decision_debate_t2", "multi_decision_debate_t3", "mdd_t3_no_veto",
        "mdd_t3_no_role",
    },
    "ollama": {
        "single_vote", "multi_vote", "multi_debate_vote_t3", "multi_decision",
        "multi_decision_debate_t3", "mdd_t3_no_veto", "mdd_t3_no_role",
    },
}


def _load_manifest(path: Path):
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                records.append({"_audit_error": f"line {line_no}: invalid JSON: {exc}"})
                continue
            records.append(record)
    return records


def _command_tokens(record):
    command = record.get("command") or record.get("cmd") or []
    if isinstance(command, str):
        return command.split()
    if isinstance(command, list):
        return [str(token) for token in command]
    return []


def _has_token(tokens, token):
    return token in tokens


def _as_int(value, default=-1):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def audit_manifest(manifest_path: Path):
    records = _load_manifest(manifest_path)
    findings = []
    for idx, record in enumerate(records, start=1):
        run_name = record.get("run_tag") or record.get("method") or f"record_{idx}"
        if "_audit_error" in record:
            findings.append((run_name, record["_audit_error"]))
            continue

        history_file = record.get("history_file") or record.get("history_path")
        if not history_file:
            findings.append((run_name, "missing history_file"))
        elif not Path(history_file).exists():
            findings.append((run_name, f"history_file not found: {history_file}"))

        for field in REQUIRED_METADATA_FIELDS:
            if field not in record or record.get(field) in (None, ""):
                findings.append((run_name, f"missing manifest metadata field: {field}"))

        tokens = _command_tokens(record)
        if not tokens:
            findings.append((run_name, "missing command/cmd field"))
        else:
            for flag, description in REQUIRED_COMMAND_FIELDS.items():
                if not _has_token(tokens, flag):
                    findings.append((run_name, f"command missing {description} ({flag})"))

        if record.get("model") not in {"qwen2.5-7b", "ollama"}:
            findings.append((run_name, f"unsupported model: {record.get('model')!r}"))
        if record.get("model") == "ollama":
            for field in (
                "ollama_model",
                "ollama_quantization",
                "ollama_model_digest",
                "ollama_version",
                "ollama_architecture",
                "ollama_parameter_count",
                "ollama_context_length",
                "ollama_embedding_length",
            ):
                if record.get(field) in (None, "", 0, "0"):
                    findings.append((run_name, f"missing Ollama deployment field: {field}"))

        method = str(record.get("method") or record.get("protocol") or "")
        model = record.get("model")
        if model in MODEL_PROTOCOLS and method not in MODEL_PROTOCOLS[model]:
            findings.append((run_name, f"protocol {method!r} is not reported for model {model!r}"))
        profile_source = record.get("profile_source")
        solver = record.get("solver")
        if solver == "decision" and profile_source != "heuristic":
            findings.append((run_name, "decision runs must use profile_source=heuristic"))
        if solver == "decision":
            try:
                epsilon_matches = abs(float(record.get("c2_epsilon")) - 0.25) <= 1e-9
            except (TypeError, ValueError):
                epsilon_matches = False
            if not epsilon_matches:
                findings.append((run_name, f"c2_epsilon={record.get('c2_epsilon')!r}; paper configuration requires 0.25"))

        for field, expected in PAPER_GENERATION_CONFIG.items():
            actual = record.get(field)
            if isinstance(expected, float):
                try:
                    matches = abs(float(actual) - expected) <= 1e-9
                except (TypeError, ValueError):
                    matches = False
            elif isinstance(expected, int) and not isinstance(expected, bool):
                try:
                    matches = int(actual) == expected
                except (TypeError, ValueError):
                    matches = False
            else:
                matches = actual == expected
            if not matches:
                findings.append((run_name, f"{field}={actual!r}; paper configuration requires {expected!r}"))

        if method in PROTOCOL_CONTROLS:
            expected_agents, expected_rounds, expected_solver, expected_roles = PROTOCOL_CONTROLS[method]
            actual_controls = (
                _as_int(record.get("num_agents")),
                _as_int(record.get("debate_rounds")),
                solver,
                bool(record.get("multi_persona")),
            )
            expected_controls = (expected_agents, expected_rounds, expected_solver, expected_roles)
            if actual_controls != expected_controls:
                findings.append((run_name, f"protocol controls {actual_controls!r} do not match {expected_controls!r}"))

        expected_veto = "not_applicable" if solver != "decision" else ("disabled" if "no_veto" in method else "enabled")
        if record.get("veto_setting") != expected_veto:
            findings.append((run_name, f"veto_setting={record.get('veto_setting')!r}; expected {expected_veto!r}"))

    return records, findings


def main():
    parser = argparse.ArgumentParser(description="Audit C2 run manifests against the reported protocol.")
    parser.add_argument("manifest", type=Path, help="Path to a JSONL manifest produced by run_c2_paper.py")
    args = parser.parse_args()

    records, findings = audit_manifest(args.manifest)
    print(f"records: {len(records)}")
    if not findings:
        print("audit: pass")
        return

    print("audit: findings")
    for run_name, message in findings:
        print(f"- {run_name}: {message}")


if __name__ == "__main__":
    main()
