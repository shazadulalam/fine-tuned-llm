
import json
import logging
import time
from typing import Dict, List, Optional
import boto3
from configs.default import SageMakerConfig

logger = logging.getLogger(__name__)

class EndpointTester:
    
    def __init__(self, config: SageMakerConfig):

        self.config = config
        session = boto3.Session(region_name=config.region)
        self.runtime = session.client("sagemaker-runtime")
        self.sm_client = session.client("sagemaker")

    def check_endpoint_status(self) -> str:

        """endpoint InService check"""
        response = self.sm_client.describe_endpoint(
            EndpointName=self.config.endpoint_name
        )
        return response["EndpointStatus"]

    def wait_for_endpoint(self, timeout: int = 600, poll_interval: int = 30) -> str:
        
        start = time.time()

        while time.time() - start < timeout:

            status = self.check_endpoint_status()
            logger.info("Endpoint status: %s", status)
            if status == "InService":
                return status
            if status in ("Failed", "RollingBack"):
                raise RuntimeError(f"Endpoint entered status: {status}")
            time.sleep(poll_interval)

        raise TimeoutError(f"Endpoint not ready after {timeout}s")

    def invoke(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
    ) -> Dict:

        """Invoke the endpoint with a prompt and return the response"""
        
        payload = {
            "inputs": f"### Instruction:\n{prompt}\n\n### Response:\n",
            "parameters": {
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "do_sample": True,
            },
        }

        response = self.runtime.invoke_endpoint(
            EndpointName=self.config.endpoint_name,
            ContentType="application/json",
            Body=json.dumps(payload),
        )

        result = json.loads(response["Body"].read().decode())

        return result

    def run_test_suite(
        self, prompts: Optional[List[str]] = None
    ) -> List[Dict]:
    
        if prompts is None:

            prompts = [
                "What is machine learning?",
                "Explain the difference between supervised and unsupervised learning.",
                "Write a Python function to sort a list.",
                "What are the benefits of fine-tuning a language model?",
                "Summarize the concept of transfer learning in NLP.",
            ]

        logger.info("Running test suite with %d prompts ...", len(prompts))
        results = []

        for i, prompt in enumerate(prompts):

            logger.info("[%d/%d] Prompt: %s", i + 1, len(prompts), prompt[:80])
            start = time.time()
            response = self.invoke(prompt)
            latency = time.time() - start
            results.append({
                "prompt": prompt,
                "response": response,
                "latency_seconds": round(latency, 3),
            })

            logger.info("  Latency: %.3fs", latency)

        return results

    def health_check(self) -> Dict:

        """Perform a basic health check on the endpoint"""
        
        status = self.check_endpoint_status()
        result = {"endpoint_name": self.config.endpoint_name, "status": status}

        if status == "InService":
            test_response = self.invoke("Hello", max_new_tokens=10)
            result["test_response"] = test_response
            result["healthy"] = True
        else:
            result["healthy"] = False

        return result