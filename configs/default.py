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


@dataclass
class TrainingConfig:
    output_dir: str = "./results"
    num_train_epochs: int = 1
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 2
    learning_rate: float = 2e-4
    optim: str = "adamw_torch"
    logging_steps: int = 5
    eval_strategy: str = "epoch"
    save_strategy: str = "epoch"
    load_best_model_at_end: bool = True
    bf16: bool = False
    fp16: bool = False
    max_seq_length: int = 256
    logging_dir: str = "./logs"


@dataclass
class DataConfig:
    dataset_name: str = "tatsu-lab/alpaca"
    instruction_col: str = "instruction"
    input_col: str = "input"
    output_col: str = "output"
    test_size: float = 0.1
    val_size: float = 0.1
    max_samples: Optional[int] = None


@dataclass
class APIConfig:
    host: str = "0.0.0.0"
    port: int = 8000
    model_path: str = "./results/final_model"
    max_new_tokens: int = 256