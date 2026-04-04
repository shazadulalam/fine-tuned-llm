import re

from datasets import Dataset

from configs.default import DataConfig


def clean_text(text: str) -> str:
    """Removing noise from text like tags, whitespace, characters"""
    
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text