from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ModelConfig:

    model_id: str = "distilbert/distilgpt2"
    load_in_4bit: bool = False
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_compute_dtype: str = "float16"
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
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    evaluation_strategy: str = "epoch"
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False
    optim: str = "adamw_torch"
    logging_steps: int = 5
    evaluation_strategy = "epoch"
    save_strategy = "epoch"
    load_best_model_at_end = True
    metric_for_best_model = "eval_loss"
    greater_is_better = False

    bf16: bool = False
    fp16: bool = True

    max_seq_length: int = 128
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
class RAGConfig:

    chunk_size: int = 512
    chunk_overlap: int = 50
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    vector_store_path: str = "./vector_store"
    top_k: int = 5
    pdf_dir: str = "./data/pdfs"
    max_new_tokens: int = 256


@dataclass
class SageMakerConfig:

    role_arn: str = ""
    instance_type: str = "ml.g5.xlarge"
    instance_count: int = 1
    model_data_s3: str = ""
    endpoint_name: str = "fine-tuned-llm-endpoint"
    region: str = "us-east-1"
    bucket: str = ""
    prefix: str = "fine-tuned-llm"
    framework_version: str = "2.1.0"
    py_version: str = "py310"
    transformers_version: str = "4.36.0"


@dataclass
class APIConfig:

    host: str = "0.0.0.0"
    port: int = 8000
    model_path: str = "./results/final_model"
    max_new_tokens: int = 256
