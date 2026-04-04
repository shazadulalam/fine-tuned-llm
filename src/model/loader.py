from typing import Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from configs.default import ModelConfig, LoraConfig


def load_base_model(config: ModelConfig) -> AutoModelForCausalLM:
    """Load base model with optional 4-bit quantization."""
    kwargs = {"device_map": "auto"}
    if config.load_in_4bit:
        kwargs["quantization_config"] = build_quantization_config(config)
    return AutoModelForCausalLM.from_pretrained(config.model_id, **kwargs)
