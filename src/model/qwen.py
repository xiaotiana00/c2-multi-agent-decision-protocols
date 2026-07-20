import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def load_model(args, checkpoint):
    if not torch.cuda.is_available():
        raise RuntimeError("The reported Qwen NF4 configuration requires a CUDA-capable GPU.")

    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        checkpoint,
        quantization_config=quantization,
        device_map="auto",
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
        token=args.token,
        cache_dir=args.model_dir,
    )
    model.eval()
    return model


class QwenWrapper:
    def __init__(self, args, checkpoint):
        self.name = checkpoint
        self.huggingface_model = load_model(args, checkpoint)
        self.tokenizer = AutoTokenizer.from_pretrained(
            checkpoint,
            token=args.token,
            cache_dir=args.model_dir,
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
