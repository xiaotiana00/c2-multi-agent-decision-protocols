import hashlib
import json
import os
import time
import urllib.error
import urllib.request

import torch

from c2_decision_utils import parse_c2_response
from c2_prompts import ROLE_PROMPTS


QWEN_CHECKPOINT = "Qwen/Qwen2.5-7B-Instruct"
DEEPSEEK_CHECKPOINT = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"


class OllamaChatWrapper:
    def __init__(self, args):
        base_url = (args.ollama_base_url or os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")).rstrip("/")
        self.name = args.ollama_model or os.getenv("OLLAMA_MODEL", "deepseek-r1:7b")
        self.chat_url = f"{base_url}/api/chat"

    def chat(self, messages, args, idx=0):
        payload = {
            "model": self.name,
            "messages": messages,
            "stream": False,
            "options": {
                "num_predict": int(args.c2_max_new_tokens),
                "temperature": float(args.c2_temperature),
                "top_p": float(args.c2_top_p),
                "top_k": int(args.c2_top_k),
                "seed": _message_seed(args.seed, messages, idx),
            },
        }
        request = urllib.request.Request(
            self.chat_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        last_error = None
        for attempt in range(3):
            try:
                with urllib.request.urlopen(request, timeout=180) as response:
                    body = json.loads(response.read().decode("utf-8"))
                return body["message"]["content"]
            except (urllib.error.URLError, TimeoutError, KeyError, json.JSONDecodeError) as exc:
                last_error = exc
                time.sleep(1.5 * (attempt + 1))
        raise RuntimeError(f"Ollama request failed at {self.chat_url}: {last_error}")


def clean_response(decoded):
    text = str(decoded or "").replace("\r\n", "\n").replace("\r", "\n")
    return "\n".join(line.strip() for line in text.splitlines() if line.strip())


def canonicalize_c2_response(decoded):
    text = clean_response(decoded)
    parsed = parse_c2_response(text)
    if not parsed["choice"]:
        return text
    reason = parsed["reason"] or "Best mission-risk tradeoff under the stated constraints."
    vetoes = sorted(parsed["vetoes"])
    veto = ",".join(vetoes) if vetoes else "NONE"
    return (
        f"{{final answer: {parsed['choice']}}}\n"
        f"Reason: {reason}\n"
        f"Confidence: {parsed['confidence']:.2f}\n"
        f"SafetyVeto: {veto}"
    )


def _message_seed(base_seed, messages, idx):
    serialized = json.dumps(messages, ensure_ascii=False, sort_keys=True)
    digest = hashlib.md5(serialized.encode("utf-8")).hexdigest()
    return int(base_seed) + idx * 997 + int(digest[:8], 16) % 100000


def _model_device(model):
    device = getattr(model, "device", None)
    if device is not None:
        return device
    return next(model.parameters()).device


def engine(messages, agent, args, num_agents=1):
    if hasattr(agent, "chat"):
        conversations = messages if messages and isinstance(messages[0], list) else [messages] * int(num_agents)
        return [canonicalize_c2_response(agent.chat(conversation, args, idx=i)) for i, conversation in enumerate(conversations)]

    if messages and isinstance(messages[0], list):
        conversations = messages
    else:
        conversations = [messages] * int(num_agents)
    prompts = [
        agent.tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
        for conversation in conversations
    ]
    encoded = agent.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
    input_ids = encoded["input_ids"].to(_model_device(agent.huggingface_model))
    attention_mask = encoded["attention_mask"].to(_model_device(agent.huggingface_model))

    outputs = []
    for index, prompt in enumerate(prompts):
        seed = _message_seed(args.seed, conversations[index], index)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        generated = agent.huggingface_model.generate(
            input_ids=input_ids[index:index + 1],
            attention_mask=attention_mask[index:index + 1],
            max_new_tokens=args.c2_max_new_tokens,
            do_sample=True,
            temperature=args.c2_temperature,
            top_p=args.c2_top_p,
            top_k=args.c2_top_k,
            use_cache=True,
            repetition_penalty=1.05,
            pad_token_id=agent.tokenizer.eos_token_id,
            eos_token_id=agent.tokenizer.eos_token_id,
            return_dict_in_generate=True,
        )
        generated_tokens = generated.sequences[0][input_ids[index].shape[-1]:]
        decoded = agent.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        outputs.append(canonicalize_c2_response(decoded))
    return outputs


def get_agents(args):
    if args.model == "qwen2.5-7b":
        from model.qwen import QwenWrapper

        if args.formal_checkpoint and args.formal_checkpoint != QWEN_CHECKPOINT:
            raise ValueError(f"Qwen runs require checkpoint {QWEN_CHECKPOINT}.")
        args.formal_checkpoint = QWEN_CHECKPOINT
        agent = QwenWrapper(args, QWEN_CHECKPOINT)
    elif args.model == "ollama":
        if args.formal_checkpoint and args.formal_checkpoint != DEEPSEEK_CHECKPOINT:
            raise ValueError(f"Ollama runs require checkpoint {DEEPSEEK_CHECKPOINT}.")
        args.formal_checkpoint = DEEPSEEK_CHECKPOINT
        agent = OllamaChatWrapper(args)
    else:
        raise ValueError(f"Unsupported model: {args.model}")

    personas = ROLE_PROMPTS if args.multi_persona else {"None": ""}
    return agent, personas
