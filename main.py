import argparse
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def run_pipeline(max_samples=None):
    
    """fine-tuning pipeline run"""

    logger.info("Pipeline initialized. Building modules...")
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

    logger.info("Data pipeline complete. Model and training modules coming next.")



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Fine-tune an LLM with QLoRA")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit dataset size for quick testing")
    args = parser.parse_args()
    run_pipeline(max_samples=args.max_samples)