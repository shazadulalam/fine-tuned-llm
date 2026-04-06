from datasets import load_dataset, Dataset, DatasetDict
from sklearn.model_selection import train_test_split

from configs.default import DataConfig


def load_hf_dataset(config: DataConfig) -> Dataset:

    """Loading dataset from HuggingFace"""
    
    dataset = load_dataset(config.dataset_name, split="train")
    if config.max_samples:
        dataset = dataset.select(range(min(config.max_samples, len(dataset))))
    return dataset

def split_dataset(dataset: Dataset, config: DataConfig) -> DatasetDict:

    """split dataset into train, validation, and test sets using stratified-friendly random split."""

    test_val_size = config.test_size + config.val_size
    val_ratio = config.val_size / test_val_size

    train_indices, test_val_indices = train_test_split(
        range(len(dataset)), test_size=test_val_size, random_state=42
    )
    val_indices, test_indices = train_test_split(
        test_val_indices, test_size=(1 - val_ratio), random_state=42
    )

    return DatasetDict({
        "train": dataset.select(train_indices),
        "validation": dataset.select(val_indices),
        "test": dataset.select(test_indices),
    })


def load_and_split(config: DataConfig) -> DatasetDict:

    """Load and spliting data into train/val/test"""

    dataset = load_hf_dataset(config)
    return split_dataset(dataset, config)