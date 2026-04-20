import math
import torch
from trl import SFTTrainer, SFTConfig
from datasets import DatasetDict

from configs.default import TrainingConfig, ModelConfig, LoraConfig
from src.model.loader import load_model_for_training, build_lora_config


def build_training_args(config: TrainingConfig) -> SFTConfig:
    
    return SFTConfig(
        output_dir=config.output_dir,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        optim=config.optim,
        logging_steps=config.logging_steps,
        eval_strategy=config.evaluation_strategy,
        save_strategy=config.save_strategy,
        load_best_model_at_end=config.load_best_model_at_end,
        bf16=config.bf16,
        fp16=config.fp16,
        max_length=config.max_seq_length,
        report_to="none",
    )

def compute_metrics(eval_pred):
    logits, labels = eval_pred

    # Convert to torch tensors
    logits = torch.tensor(logits)
    labels = torch.tensor(labels)

    # Shift for causal language modeling
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
    loss = loss_fct(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1)
    ).item()

    perplexity = math.exp(loss) if loss < 20 else float("inf")

    return {
        "perplexity": perplexity
    }


def create_trainer(
    model_config: ModelConfig,
    lora_config: LoraConfig,
    training_config: TrainingConfig,
    datasets: DatasetDict,
) -> SFTTrainer:

    """SFTTrainer with model, data, and training configuration"""

    model, tokenizer = load_model_for_training(model_config, lora_config)
    sft_config = build_training_args(training_config)

    trainer = SFTTrainer(
        model=model,
        train_dataset=datasets["train"],
        eval_dataset=datasets["validation"],
        args=sft_config,
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
    )

    return trainer


def train_and_save(trainer: SFTTrainer, output_path: str) -> dict:

    """fine-tuned model training in loop and evaluation results"""
    
    trainer.train()
    eval_results = trainer.evaluate()
    trainer.save_model(output_path)
    return eval_results
