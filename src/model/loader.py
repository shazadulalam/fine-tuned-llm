from typing import Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from configs.default import ModelConfig, LoraConfig

DTYPE_MAP = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}

def load_tokenizer(config: ModelConfig) -> AutoTokenizer:

    """tokenizer loading and configure with padding token"""

    tokenizer = AutoTokenizer.from_pretrained(config.model_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    return tokenizer

def build_quantization_config(config: ModelConfig) -> BitsAndBytesConfig:

    """QLoRA quantization config build"""

    return BitsAndBytesConfig(
        load_in_4bit=config.load_in_4bit,
        bnb_4bit_quant_type=config.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=DTYPE_MAP[config.bnb_4bit_compute_dtype],
        bnb_4bit_use_double_quant=config.use_double_quant,
    )

def load_base_model(config: ModelConfig) -> AutoModelForCausalLM:

    """load base model with optional 4-bit quantization"""

    kwargs = {"device_map": "auto"}
    if config.load_in_4bit:
        kwargs["quantization_config"] = build_quantization_config(config)

    return AutoModelForCausalLM.from_pretrained(config.model_id, **kwargs)


def build_lora_config(config: LoraConfig) -> PeftLoraConfig:

    """building PEFT LoRA configuration"""

    return PeftLoraConfig(
        r=config.r,
        lora_alpha=config.lora_alpha,
        target_modules=config.target_modules,
        lora_dropout=config.lora_dropout,
        bias=config.bias,
        task_type=config.task_type,
    )


def load_model_for_training(
    model_config: ModelConfig, lora_config: LoraConfig
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:

    """loading model with QLoRA adapters ready for fine-tuning.

    Returns (model, tokenizer) tuple.
    """
    
    tokenizer = load_tokenizer(model_config)
    model = load_base_model(model_config)

    if model_config.load_in_4bit:
        model = prepare_model_for_kbit_training(model)

    peft_config = build_lora_config(lora_config)
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    return model, tokenizer
