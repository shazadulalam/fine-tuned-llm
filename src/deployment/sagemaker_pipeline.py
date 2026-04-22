import os
import json
import tarfile
import logging
from typing import Optional

import boto3
import sagemaker
from sagemaker.huggingface import HuggingFaceModel
from sagemaker.session import Session

from configs.default import SageMakerConfig

logger = logging.getLogger(__name__)


def package_model(model_dir: str, output_path: str = "model.tar.gz") -> str:

    """model artifacts convert into a tar.gz for SageMaker deployment"""

    logger.info("Model packaging from %s into %s ...", model_dir, output_path)

    with tarfile.open(output_path, "w:gz") as tar:

        for root, _, files in os.walk(model_dir):
            for f in files:
                filepath = os.path.join(root, f)
                arcname = os.path.relpath(filepath, model_dir)
                tar.add(filepath, arcname=arcname)

    logger.info("Model packaged: %s", output_path)

    return output_path


def upload_model_to_s3(
    tarball_path: str, config: SageMakerConfig
) -> str:

    """Packaged model upload to S3 and return the S3 URI"""

    session = boto3.Session(region_name=config.region)
    s3 = session.client("s3")
    s3_key = f"{config.prefix}/model.tar.gz"

    logger.info("Uploading %s to s3://%s/%s ...", tarball_path, config.bucket, s3_key)
    s3.upload_file(tarball_path, config.bucket, s3_key)

    s3_uri = f"s3://{config.bucket}/{s3_key}"
    logger.info("Upload complete: %s", s3_uri)

    return s3_uri


def create_sagemaker_model(config: SageMakerConfig, model_data_s3: str) -> HuggingFaceModel:

    """SageMaker HuggingFace model object creation"""

    hub_env = {
        "HF_TASK": "text-generation",
        "SM_NUM_GPUS": "1",
    }
    return HuggingFaceModel(
        model_data=model_data_s3,
        role=config.role_arn,
        transformers_version=config.transformers_version,
        pytorch_version=config.framework_version,
        py_version=config.py_version,
        env=hub_env,
    )


def deploy_endpoint(config: SageMakerConfig, model_data_s3: Optional[str] = None) -> str:

    """Deploying the model to a SageMaker real-time endpoint and returns the endpoint name"""

    s3_uri = model_data_s3 or config.model_data_s3
    logger.info("Deploying endpoint '%s' with instance %s ...",
                config.endpoint_name, config.instance_type)

    hf_model = create_sagemaker_model(config, s3_uri)
    predictor = hf_model.deploy(
        initial_instance_count=config.instance_count,
        instance_type=config.instance_type,
        endpoint_name=config.endpoint_name,
    )

    logger.info("Endpoint '%s' is now InService.", config.endpoint_name)

    return config.endpoint_name


def delete_endpoint(config: SageMakerConfig) -> None:

    """delete the SageMaker endpoint """

    session = boto3.Session(region_name=config.region)
    sm_client = session.client("sagemaker")

    logger.info("Deleting endpoint '%s' ...", config.endpoint_name)
    sm_client.delete_endpoint(EndpointName=config.endpoint_name)
    logger.info("Endpoint deleted.")


def run_sagemaker_pipeline(
    model_dir: str,
    config: SageMakerConfig,
) -> str:

    """package model -> upload to S3 -> deploy endpoint"""

    tarball = package_model(model_dir)
    s3_uri = upload_model_to_s3(tarball, config)
    endpoint_name = deploy_endpoint(config, s3_uri)

    # previous tar file remove
    if os.path.exists(tarball):
        os.remove(tarball)

    return endpoint_name