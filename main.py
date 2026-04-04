import argparse
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def run_pipeline(max_samples=None):
    
    """fine-tuning pipeline run"""
    logger.info("Pipeline initialized. Building modules...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune an LLM with QLoRA")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit dataset size for quick testing")
    args = parser.parse_args()
    run_pipeline(max_samples=args.max_samples)