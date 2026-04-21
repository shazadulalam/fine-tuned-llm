import argparse
import json
import logging

from configs.default import DataConfig, ModelConfig, LoraConfig, TrainingConfig
from src.data.loader import load_and_split
from src.data.preprocessing import preprocess_dataset, deduplicate
from src.training.trainer import create_trainer, train_and_save
from configs.default import RAGConfig, ModelConfig
from src.rag.pipeline import run_rag_pipeline, ingest_pdfs
from configs.default import RAGConfig
from src.rag.pipeline import ingest_pdfs
from configs.default import SageMakerConfig
from src.deployment.sagemaker_pipeline import run_sagemaker_pipeline
from configs.default import SageMakerConfig
from src.deployment.endpoint_tester import EndpointTester


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

    logger.info("Preprocessing and remove duplicate...")
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

    for key, val in eval_results.items():
        logger.info("  %-25s : %.4f", key, val)

    # model evaluation on test set 

    test_texts = datasets["test"]["text"]
    test_prompts = [t.split("### Response:")[0] + "### Response:\n" for t in test_texts]
    test_references = [t.split("### Response:\n")[-1] for t in test_texts]

    num_eval = min(50, len(test_texts))
    metrics = evaluate_model(
        trainer.model,
        trainer.processing_class,
        test_texts[:num_eval],
        test_prompts[:num_eval],
        test_references[:num_eval],
    )

    logger.info("  Perplexity : %.2f", metrics["perplexity"])
    logger.info("  ROUGE-1    : %.4f", metrics["rouge1"])
    logger.info("  ROUGE-2    : %.4f", metrics["rouge2"])
    logger.info("  ROUGE-L    : %.4f", metrics["rougeL"])
    logger.info("=" * 60)
    logger.info("Model saved to: %s", output_path)

    results_file = f"{training_config.output_dir}/results.json"
    with open(results_file, "w") as f:
        json.dump({**eval_results, **metrics}, f, indent=2)
    logger.info("Results saved to: %s", results_file)

    return metrics

def run_rag(pdf_dir=None, queries=None, model_path=None):

    """PDF ingestion and answer queries."""

    rag_cfg = RAGConfig()
    model_cfg = ModelConfig()

    if pdf_dir:
        rag_cfg.pdf_dir = pdf_dir
    if model_path:
        model_path_resolved = model_path
    else:
        model_path_resolved = "./results/final_model"

    if queries is None:
        queries = ["What is the main topic of the document?"]

    logger.info("PDF directory    : %s", rag_cfg.pdf_dir)
    logger.info("Embedding model  : %s", rag_cfg.embedding_model)
    logger.info("Chunk size       : %d", rag_cfg.chunk_size)
    logger.info("Top-K retrieval  : %d", rag_cfg.top_k)
    logger.info("=" * 60)

    results = run_rag_pipeline(rag_cfg, model_cfg, model_path_resolved, queries)

    for r in results:

        logger.info("Sources: %s", [s["source"] for s in r["sources"]])

    return results

def run_ingest(pdf_dir=None):

    rag_cfg = RAGConfig()
    if pdf_dir:
        rag_cfg.pdf_dir = pdf_dir

    logger.info("Ingesting PDFs from %s ...", rag_cfg.pdf_dir)
    store = ingest_pdfs(rag_cfg)
    logger.info("Ingestion complete. %d vectors in store.", store.total_documents)
    
    return store


def run_deploy(model_dir=None):

    """Model deployment to AWS SageMaker"""

    sm_cfg = SageMakerConfig()
    model_dir = model_dir or "./results/final_model"

    endpoint_name = run_sagemaker_pipeline(model_dir, sm_cfg)
    logger.info("Endpoint live: %s", endpoint_name)
    return endpoint_name


def run_test_endpoint():

    """sageMaker endpoint test"""

    sm_cfg = SageMakerConfig()
    tester = EndpointTester(sm_cfg)

    health = tester.health_check()
    logger.info("Health check: %s", json.dumps(health, indent=2, default=str))

    results = tester.run_test_suite()
    for r in results:
        
        logger.info("Prompt  : %s", r["prompt"][:80])
        logger.info("Latency : %.3fs", r["latency_seconds"])
        logger.info("Response: %s", str(r["response"])[:200])

    return results


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Fine-tune LLM with QLoRA, RAG, and SageMaker")
    subparsers = parser.add_subparsers(dest="command", help="Pipeline to run")

    # Train
    train_parser = subparsers.add_parser("train", help="Fine-tune the model")
    train_parser.add_argument("--max-samples", type=int, default=None)

    # RAG
    rag_parser = subparsers.add_parser("rag", help="Run RAG pipeline")
    rag_parser.add_argument("pdf-dir", type=str, default=None)
    rag_parser.add_argument("model-path", type=str, default=None)
    rag_parser.add_argument("query", type=str, nargs="+", default=None)

    # Ingest
    ingest_parser = subparsers.add_parser("ingest", help="Ingest PDFs into vector store")
    ingest_parser.add_argument("pdf-dir", type=str, default=None)

    # Deploy
    deploy_parser = subparsers.add_parser("deploy", help="Deploy to SageMaker")
    deploy_parser.add_argument("model-dir", type=str, default=None)

    # Test endpoint
    subparsers.add_parser("test-endpoint", help="Test live SageMaker endpoint")

    args = parser.parse_args()

    if args.command == "train":
        run_pipeline(max_samples=args.max_samples)
    elif args.command == "rag":
        run_rag(pdf_dir=args.pdf_dir, queries=args.query, model_path=args.model_path)
    elif args.command == "ingest":
        run_ingest(pdf_dir=args.pdf_dir)
    elif args.command == "deploy":
        run_deploy(model_dir=args.model_dir)
    elif args.command == "test-endpoint":
        run_test_endpoint()
    else:
        run_pipeline()