import re

from datasets import Dataset

from configs.default import DataConfig


def clean_text(text: str) -> str:
    """Removing noise from text like tags, whitespace, characters"""
    
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def format_instruction(example: dict, config: DataConfig) -> dict:

    """formating the dataset row into instruction-response chat format"""

    instruction = clean_text(example.get(config.instruction_col, ""))
    input_text = clean_text(example.get(config.input_col, ""))
    output_text = clean_text(example.get(config.output_col, ""))

    prompt = f"{instruction}\n{input_text}".strip() if input_text else instruction
    text = f"### Instruction:\n{prompt}\n\n### Response:\n{output_text}"
    return {"text": text}


def preprocess_dataset(dataset: Dataset, config: DataConfig) -> Dataset:

    """cleaning and formatting to entire dataset"""

    return dataset.map(
        lambda ex: format_instruction(ex, config),
        remove_columns=dataset.column_names,
    )


def deduplicate(dataset: Dataset) -> Dataset:

    """duplicate entries based on text content removal"""

    seen = set()
    unique_indices = [
        i for i, text in enumerate(dataset["text"])
        if text not in seen and not seen.add(text)
    ]
    return dataset.select(unique_indices)