import math
from typing import Dict, List

import torch
from rouge_score import rouge_scorer
from transformers import AutoModelForCausalLM, AutoTokenizer


def compute_perplexity(model: AutoModelForCausalLM, tokenizer: AutoTokenizer, texts: List[str]) -> float:
    
    """average perplexity over a list of texts calculated"""

    model.eval()
    total_loss, total_tokens = 0.0, 0

    for text in texts:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024).to(model.device)
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])
        total_loss += outputs.loss.item() * inputs["input_ids"].shape[1]
        total_tokens += inputs["input_ids"].shape[1]

    return math.exp(total_loss / total_tokens) if total_tokens > 0 else float("inf")


def compute_rouge(predictions: List[str], references: List[str]) -> Dict[str, float]:

    """ROUGE-1, ROUGE-2, and ROUGE-L F1 scores calculation"""

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    scores = {"rouge1": [], "rouge2": [], "rougeL": []}

    for pred, ref in zip(predictions, references):
        result = scorer.score(ref, pred)
        for key in scores:
            scores[key].append(result[key].fmeasure)

    return {key: sum(vals) / len(vals) for key, vals in scores.items()}


def evaluate_model(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    test_texts: List[str],
    test_prompts: List[str],
    test_references: List[str],
    max_new_tokens: int = 256,
) -> Dict:

    """full evaluation train: perplexity + ROUGE scores"""
    
    perplexity = compute_perplexity(model, tokenizer, test_texts)

    predictions = []
    model.eval()
    for prompt in test_prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            output_ids = model.generate(
                **inputs, max_new_tokens=max_new_tokens, pad_token_id=tokenizer.eos_token_id
            )
        generated = output_ids[0][inputs["input_ids"].shape[1]:]
        predictions.append(tokenizer.decode(generated, skip_special_tokens=True))

    rouge = compute_rouge(predictions, test_references)

    return {"perplexity": perplexity, **rouge}