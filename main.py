import argparse
import json
import logging

from configs.default import DataConfig, ModelConfig, LoraConfig, TrainingConfig
from src.data.loader import load_and_split
from src.data.preprocessing import preprocess_dataset, deduplicate
from src.training.trainer import create_trainer, train_and_save

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def run_pipeline(max_samples=None):

    """complete fine-tuning pipeline run"""
    
    
    logger.info("Loading and splitting dataset...")
    data_config = DataConfig(max_samples=max_samples)
    datasets = load_and_split(data_config)
    logger.info(
        "Dataset sizes — train: %d, val: %d, test: %d",
        len(datasets["train"]), len(datasets["validation"]), len(datasets["test"]),
    )

    logger.info("Preprocessing and deduplicating...")
    from datasets import DatasetDict
    processed = {}
    for split_name in datasets:
        processed[split_name] = preprocess_dataset(datasets[split_name], data_config)
        processed[split_name] = deduplicate(processed[split_name])
    datasets = DatasetDict(processed)

    logger.info("After dedup — train: %d, val: %d, test: %d",
        len(datasets["train"]), len(datasets["validation"]), len(datasets["test"]),
    )

    # model training
    
    logger.info("Initializing model and trainer...")
    model_config = ModelConfig()
    lora_config = LoraConfig()
    training_config = TrainingConfig()

    trainer = create_trainer(model_config, lora_config, training_config, datasets)

    logger.info("Starting fine-tuning...")
    output_path = f"{training_config.output_dir}/final_model"
    eval_results = train_and_save(trainer, output_path)

    logger.info("Training complete. Loss: %s", json.dumps(eval_results, indent=2))
    logger.info("Model saved to %s", output_path)

    return eval_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune an LLM with QLoRA")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit dataset size for quick testing")
    args = parser.parse_args()
    run_pipeline(max_samples=args.max_samples)