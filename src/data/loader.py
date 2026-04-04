from datasets import load_dataset, Dataset, DatasetDict
from sklearn.model_selection import train_test_split

from configs.default import DataConfig


def load_hf_dataset(config: DataConfig) -> Dataset:
    """Loading dataset from HuggingFace"""
    dataset = load_dataset(config.dataset_name, split="train")
    if config.max_samples:
        dataset = dataset.select(range(min(config.max_samples, len(dataset))))
    return dataset