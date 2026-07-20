import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import random
import re
import time

import numpy as np
import torch
from tqdm import tqdm

from c2_decision_utils import parse_c2_response
from c2_prompts import C2_DEBATE_RULE, C2_SYSTEM_PROMPT, render_sample
from data.data_utils import load_data
from evaluator import evaluate_c2, get_instruction_suffix
from model.model_utils import engine, get_agents


PAPER_EPSILON = 0.25
PAPER_TEMPERATURE = 0.14
PAPER_TOP_P = 0.78
PAPER_TOP_K = 8
PAPER_MAX_NEW_TOKENS = 64


def _hf_token():
    return os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")


def _summarize_response(response, include_private_fields=True):
    parsed = parse_c2_response(response)
    parts = [f"Choice={parsed['choice'] or 'NO_VALID_CHOICE'}"]
    if parsed["reason"]:
        parts.append(f"Reason={parsed['reason']}")
    if include_private_fields:
        parts.append(f"Confidence={parsed['confidence']:.2f}")
        if parsed["vetoes"]:
            parts.append(f"SafetyVeto={','.join(sorted(parsed['vetoes']))}")
    return "; ".join(parts)


def _sample_text(args, sample, role=None):
    if args.multi_persona and role:
        return render_sample(sample, persona_key=role)
    return render_sample(sample)


def get_new_message(args, sample, responses, personas=None, suffix=None):
    messages = {}
    names = list(responses)
    for name in names:
        role = name.split("__")[-2] if args.multi_persona and "__" in name else None
        lines = [C2_DEBATE_RULE, "Other agents' previous choices and reasons:"]
        for other_name in names:
            if other_name != name:
                lines.append(f"- {other_name}: {_summarize_response(responses[other_name], False)}")
        lines.append(f"Your previous response: {_summarize_response(responses[name], True)}")
        if role:
            lines.append("Reconsider the shared scenario using your original private note:")
        else:
            lines.append("Reconsider the shared scenario:")
        lines.append(_sample_text(args, sample, role=role))
        user_message = "\n".join(lines)
        if role:
            messages[name] = [
                {"role": "system", "content": personas[role]},
                {"role": "user", "content": user_message},
            ]
        else:
            messages[name] = {"role": "user", "content": user_message}
    return messages


def _package_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _runtime_environment(args):
    gpu_model = None
    gpu_memory_gib = None
    if torch.cuda.is_available():
        properties = torch.cuda.get_device_properties(0)
        gpu_model = properties.name
        gpu_memory_gib = round(properties.total_memory / (1024 ** 3), 2)

    if args.model == "qwen2.5-7b":
        backend = "huggingface_transformers"
        quantization = {
            "format": "NF4",
            "load_in_4bit": True,
            "double_quantization": True,
            "compute_dtype": "FP16",
        }
    else:
        backend = "ollama_native"
        quantization = {
            "format": args.ollama_quantization or None,
            "ollama_tag": args.ollama_model,
            "ollama_model_digest": args.ollama_model_digest or None,
        }

    return {
        "inference_backend": backend,
        "python_version": platform.python_version(),
        "operating_system": platform.platform(),
        "torch_version": torch.__version__,
        "transformers_version": _package_version("transformers"),
        "accelerate_version": _package_version("accelerate"),
        "bitsandbytes_version": _package_version("bitsandbytes"),
        "cuda_runtime": torch.version.cuda,
        "gpu_model": gpu_model,
        "dedicated_gpu_memory_gib": gpu_memory_gib,
        "quantization": quantization,
        "ollama_version": args.ollama_version or None,
        "ollama_architecture": args.ollama_architecture or None,
        "ollama_parameter_count": args.ollama_parameter_count or None,
        "ollama_context_length": args.ollama_context_length or None,
        "ollama_embedding_length": args.ollama_embedding_length or None,
    }


def get_args():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_dir", default="out")
    parser.add_argument("--data", choices=["c2_decision"], default="c2_decision")
    parser.add_argument("--data_dir", default=None)
    parser.add_argument("--data_size", type=int, default=100)
    parser.add_argument("--c2_data_file", default=None)
    parser.add_argument("--c2_version", choices=["v4"], default="v4")
    parser.add_argument("--model", choices=["qwen2.5-7b", "ollama"], default="qwen2.5-7b")
    parser.add_argument("--model_dir", default=None)
    parser.add_argument("--formal_checkpoint", default="")
    parser.add_argument("--num_agents", type=int, choices=[1, 3], default=3)
    parser.add_argument("--debate_rounds", type=int, choices=[0, 1, 2, 3], default=0)
    parser.add_argument("--solver", choices=["vote", "decision"], default="vote")
    parser.add_argument("--multi_persona", action="store_true")
    parser.add_argument("--c2_disable_safety_veto", action="store_true")
    parser.add_argument("--c2_epsilon", type=float, choices=[PAPER_EPSILON], default=PAPER_EPSILON)
    parser.add_argument("--c2_temperature", type=float, choices=[PAPER_TEMPERATURE], default=PAPER_TEMPERATURE)
    parser.add_argument("--c2_top_p", type=float, choices=[PAPER_TOP_P], default=PAPER_TOP_P)
    parser.add_argument("--c2_top_k", type=int, choices=[PAPER_TOP_K], default=PAPER_TOP_K)
    parser.add_argument(
        "--c2_max_new_tokens",
        type=int,
        choices=[PAPER_MAX_NEW_TOKENS],
        default=PAPER_MAX_NEW_TOKENS,
    )
    parser.add_argument("--run_tag", default="")
    parser.add_argument("--ollama_base_url", default=os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434"))
    parser.add_argument("--ollama_model", default=os.getenv("OLLAMA_MODEL", "deepseek-r1:7b"))
    parser.add_argument("--ollama_quantization", default=os.getenv("OLLAMA_QUANTIZATION", ""))
    parser.add_argument("--ollama_model_digest", default=os.getenv("OLLAMA_MODEL_DIGEST", ""))
    parser.add_argument("--ollama_version", default=os.getenv("OLLAMA_VERSION", ""))
    parser.add_argument("--ollama_architecture", default=os.getenv("OLLAMA_ARCHITECTURE", ""))
    parser.add_argument("--ollama_parameter_count", default=os.getenv("OLLAMA_PARAMETER_COUNT", ""))
    parser.add_argument("--ollama_context_length", type=int, default=int(os.getenv("OLLAMA_CONTEXT_LENGTH", "0") or 0))
    parser.add_argument("--ollama_embedding_length", type=int, default=int(os.getenv("OLLAMA_EMBEDDING_LENGTH", "0") or 0))
    return parser.parse_args()


def _build_initial_messages(args, sample, personas, persona_keys, suffix):
    if args.multi_persona:
        messages = []
        for index in range(args.num_agents):
            role = persona_keys[index % len(persona_keys)]
            messages.append([
                {"role": "system", "content": f"{personas[role]}\n{C2_SYSTEM_PROMPT}"},
                {"role": "user", "content": f"{_sample_text(args, sample, role)}{suffix}"},
            ])
        return messages
    return [
        {"role": "system", "content": C2_SYSTEM_PROMPT},
        {"role": "user", "content": f"{_sample_text(args, sample)}{suffix}"},
    ]


def _round_boolean_rates(rows):
    if not rows:
        return []
    return [
        float(np.mean([bool(row[index]) for row in rows if len(row) > index]))
        for index in range(max(map(len, rows)))
    ]


def _round_mean(rows):
    if not rows:
        return []
    return [
        float(np.mean([float(row[index]) for row in rows if len(row) > index]))
        for index in range(max(map(len, rows)))
    ]


def _answer_change_rate(previous, current):
    if not previous or not current or len(previous) != len(current):
        return 0.0
    valid_pairs = [(a, b) for a, b in zip(previous, current) if a and b]
    if not valid_pairs:
        return 0.0
    return sum(a != b for a, b in valid_pairs) / len(valid_pairs)


def _disagreement_rate(answers):
    valid = [answer for answer in answers if answer]
    return float(len(valid) > 1 and len(set(valid)) > 1)


def _sample_option(sample, field):
    value = sample.get(field)
    return f"({str(value).strip().upper()})" if value else None


def _evaluate_round(args, sample, answer, responses, round_index):
    return evaluate_c2(
        responses,
        answer,
        sample,
        solver=args.solver,
        epsilon=args.c2_epsilon,
        safety_veto=not args.c2_disable_safety_veto,
        debate_mode=round_index > 0,
    )


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, set):
        return sorted(value)
    if isinstance(value, tuple):
        return list(value)
    raise TypeError(f"Cannot serialize {type(value)}")


def _run_config(args, personas):
    runtime = _runtime_environment(args)
    return {
        "seed": args.seed,
        "data": args.data,
        "data_size": args.data_size,
        "c2_version": args.c2_version,
        "c2_loaded_file": getattr(args, "_c2_loaded_file", None),
        "c2_dataset_sha256": getattr(args, "_c2_dataset_sha256", None),
        "c2_dataset_metadata_file": getattr(args, "_c2_dataset_metadata_file", None),
        "model": args.model,
        "formal_checkpoint": args.formal_checkpoint,
        "inference_backend": runtime["inference_backend"],
        "runtime_environment": runtime,
        "ollama_model": args.ollama_model if args.model == "ollama" else None,
        "ollama_base_url": args.ollama_base_url if args.model == "ollama" else None,
        "ollama_quantization": args.ollama_quantization if args.model == "ollama" else None,
        "ollama_model_digest": args.ollama_model_digest if args.model == "ollama" else None,
        "ollama_version": args.ollama_version if args.model == "ollama" else None,
        "ollama_architecture": args.ollama_architecture if args.model == "ollama" else None,
        "ollama_parameter_count": args.ollama_parameter_count if args.model == "ollama" else None,
        "ollama_context_length": args.ollama_context_length if args.model == "ollama" else None,
        "ollama_embedding_length": args.ollama_embedding_length if args.model == "ollama" else None,
        "num_agents": args.num_agents,
        "debate_rounds": args.debate_rounds,
        "solver": args.solver,
        "multi_persona": args.multi_persona,
        "do_sample": True,
        "temperature": args.c2_temperature,
        "top_p": args.c2_top_p,
        "top_k": args.c2_top_k,
        "max_new_tokens": args.c2_max_new_tokens,
        "c2_profile_source": "heuristic" if args.solver == "decision" else "not_applicable",
        "c2_epsilon": args.c2_epsilon,
        "c2_safety_veto": not args.c2_disable_safety_veto,
        "prompt_version": "c2_prompt_v2",
        "prompt_hash": hashlib.sha256(
            (C2_SYSTEM_PROMPT + C2_DEBATE_RULE + json.dumps(personas, sort_keys=True)).encode("utf-8")
        ).hexdigest(),
        "run_tag": args.run_tag,
    }


def main(args):
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    args.token = _hf_token()

    os.makedirs(os.path.join(args.out_dir, "history"), exist_ok=True)
    agent, personas = get_agents(args)
    samples, labels = load_data(args, split="test")
    if not samples:
        raise ValueError("The C2 dataset is empty.")

    persona_keys = list(personas)
    agent_names = [
        f"{args.model}__{persona_keys[index % len(persona_keys)] if args.multi_persona else 'None'}__Agent{index + 1}"
        for index in range(args.num_agents)
    ]
    config = _run_config(args, personas)
    histories = []
    correctness, risk_hits, catastrophic_hits = [], [], []
    disagreement_rates, answer_change_rates = [], []
    started = time.time()

    for sample_index, (sample, answer) in tqdm(enumerate(zip(samples, labels)), total=len(samples)):
        suffix = get_instruction_suffix(args)
        messages = _build_initial_messages(args, sample, personas, persona_keys, suffix)
        generated = engine(messages, agent, args, args.num_agents)
        responses = dict(zip(agent_names, generated))
        round_records = {}
        sample_correctness, sample_risk, sample_catastrophic = [], [], []
        sample_disagreement, sample_changes = [], []
        previous_answers = None
        risk_option = _sample_option(sample, "risk_option")
        catastrophic_option = _sample_option(sample, "catastrophic_option")

        for round_index in range(args.debate_rounds + 1):
            if round_index > 0:
                debate_messages = get_new_message(args, sample, responses, personas if args.multi_persona else None)
                message_batch = list(debate_messages.values()) if args.multi_persona else [[message] for message in debate_messages.values()]
                generated = engine(message_batch, agent, args, args.num_agents)
                responses = dict(zip(agent_names, generated))

            final_answers, aggregate_answer, is_correct, diagnostics = _evaluate_round(
                args, sample, answer, responses, round_index
            )
            risk_hit = bool(risk_option and aggregate_answer == risk_option)
            catastrophic_hit = bool(catastrophic_option and aggregate_answer == catastrophic_option)
            disagreement = _disagreement_rate(final_answers)
            change_rate = _answer_change_rate(previous_answers, final_answers) if previous_answers is not None else 0.0
            round_records[str(round_index)] = {
                "responses": responses,
                "parsed_responses": {name: parse_c2_response(text) for name, text in responses.items()},
                "final_answers": final_answers,
                "final_answer_iscorr": [prediction == answer for prediction in final_answers],
                "aggregate_answer": aggregate_answer,
                "aggregate_answer_iscorr": bool(is_correct),
                "answer": answer,
                "risk_option": risk_option,
                "risk_hit": risk_hit,
                "catastrophic_option": catastrophic_option,
                "catastrophic_hit": catastrophic_hit,
                "disagreement_rate": disagreement,
                "answer_change_rate": change_rate,
                "aggregator_diagnostics": diagnostics,
            }
            sample_correctness.append(bool(is_correct))
            sample_risk.append(risk_hit)
            sample_catastrophic.append(catastrophic_hit)
            sample_disagreement.append(disagreement)
            sample_changes.append(change_rate)
            previous_answers = final_answers

        histories.append({
            "sample_id": sample["id"],
            "sample_index": sample_index,
            "task_type": sample.get("task_type"),
            "sample": sample,
            "answer": answer,
            "rounds": round_records,
            "run_config": config,
        })
        correctness.append(sample_correctness)
        risk_hits.append(sample_risk)
        catastrophic_hits.append(sample_catastrophic)
        disagreement_rates.append(sample_disagreement)
        answer_change_rates.append(sample_changes)

    method_name = f"{args.data}_{len(samples)}__{args.model}_N={args.num_agents}_R={args.debate_rounds}_S={args.solver}"
    if args.solver == "decision":
        method_name += "_P=heuristic_E=0p25"
    if args.multi_persona:
        method_name += "_HETERO"
    if args.run_tag:
        method_name += f"__{re.sub(r'[^A-Za-z0-9_.-]+', '-', args.run_tag)}"

    history_path = os.path.join(args.out_dir, "history", f"{method_name}.jsonl")
    with open(history_path, "w", encoding="utf-8") as handle:
        for item in histories:
            handle.write(json.dumps(item, ensure_ascii=False, default=_json_default) + "\n")

    metrics = {
        "round_accs": _round_boolean_rates(correctness),
        "risk_rates": _round_boolean_rates(risk_hits),
        "catastrophic_rates": _round_boolean_rates(catastrophic_hits),
        "disagreement_rates": _round_mean(disagreement_rates),
        "answer_change_rates": _round_mean(answer_change_rates),
    }
    log_path = os.path.join(args.out_dir, "logs.tsv")
    write_header = not os.path.exists(log_path)
    with open(log_path, "a", encoding="utf-8") as handle:
        if write_header:
            handle.write("timestamp\tmethod\tseed\tmetrics\telapsed_sec\n")
        handle.write(
            f"{time.strftime('%Y-%m-%d %H:%M:%S')}\t{method_name}\t{args.seed}\t"
            f"{json.dumps(metrics)}\t{time.time() - started:.2f}\n"
        )
    print(json.dumps({"history": history_path, **metrics}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main(get_args())
