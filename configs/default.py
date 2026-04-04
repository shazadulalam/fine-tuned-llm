from dataclasses import dataclass, field
from typing import List, Optional


@datasclass
def ModelConfig:

    model_id: str = "distilbert/distilgp2"
    load_in_4bit: bool = False
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_compute_type: str = "float16"
    use_double_quant: bool = True


@dataclass
class LoraConfig:
    r: int = 16
    lora_alpha: int = 32
    target_modules: List[str] = field(
        default_factory=lambda: [
            "c_attn", "c_proj",
        ]
    )
    lora_dropout: float = 0.05
    bias: str = "none"
    task_type: str = "CAUSAL_LM"