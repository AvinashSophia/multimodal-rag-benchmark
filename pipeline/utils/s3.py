"""S3 utility for the benchmarking pipeline.

Thin boto3 wrapper used by AWS-variant components for uploading outputs,
downloading datasets, and accessing images from the staging bucket.

Bucket: spatial-ai-staging-processing-632872792182
Region: us-west-2

S3 layout:
    benchmarking/datasets/   ← input datasets and corpora
    benchmarking/images/     ← page images for visual retrieval
    benchmarking/results/    ← benchmark run outputs (config, results, metrics)
    benchmarking/feedback/   ← human evaluation feedback

Usage:
    from pipeline.utils.s3 import S3Client
    s3 = S3Client(config)
    s3.upload_file("results.json", "benchmarking/results/run_001/results.json")
    s3.download_file("benchmarking/datasets/altumint/qa_pairs.json", "local.json")
"""

import json
import os
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import boto3
from botocore.exceptions import ClientError


class S3Client:
    """Thin boto3 wrapper configured from pipeline config or environment.

    Config keys (under s3):
        bucket:   S3 bucket name (default: spatial-ai-staging-processing-632872792182)
        region:   AWS region      (default: us-west-2)
        prefixes.datasets:  Prefix for datasets  (default: benchmarking/datasets)
        prefixes.images:    Prefix for images     (default: benchmarking/images)
        prefixes.results:   Prefix for results    (default: benchmarking/results)
        prefixes.feedback:  Prefix for feedback   (default: benchmarking/feedback)
    """

    DEFAULT_BUCKET = "spatial-ai-staging-processing-632872792182"
    DEFAULT_REGION = "us-west-2"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        s3_cfg = (config or {}).get("s3", {})
        self.bucket = s3_cfg.get("bucket", self.DEFAULT_BUCKET)
        region = s3_cfg.get("region", self.DEFAULT_REGION)

        prefixes = s3_cfg.get("prefixes", {})
        self.prefix_datasets = prefixes.get("datasets", "benchmarking/datasets")
        self.prefix_images   = prefixes.get("images",   "benchmarking/images")
        self.prefix_results  = prefixes.get("results",  "benchmarking/results")
        self.prefix_feedback = prefixes.get("feedback", "benchmarking/feedback")

        profile = os.environ.get("AWS_PROFILE")
        session = boto3.Session(profile_name=profile, region_name=region)
        self._client = session.client("s3")

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    def upload_file(self, local_path: Union[str, Path], s3_key: str) -> None:
        """Upload a local file to S3."""
        self._client.upload_file(str(local_path), self.bucket, s3_key)

    def download_file(self, s3_key: str, local_path: Union[str, Path]) -> None:
        """Download an S3 object to a local file. Creates parent dirs as needed."""
        Path(local_path).parent.mkdir(parents=True, exist_ok=True)
        self._client.download_file(self.bucket, s3_key, str(local_path))

    def upload_json(self, data: Any, s3_key: str) -> None:
        """Serialize data as JSON and upload directly to S3 (no temp file)."""
        body = json.dumps(data, indent=2, default=str).encode("utf-8")
        self._client.put_object(
            Bucket=self.bucket,
            Key=s3_key,
            Body=body,
            ContentType="application/json",
        )

    def download_json(self, s3_key: str) -> Any:
        """Download and deserialize a JSON object from S3."""
        response = self._client.get_object(Bucket=self.bucket, Key=s3_key)
        return json.loads(response["Body"].read().decode("utf-8"))

    def upload_bytes(self, data: bytes, s3_key: str, content_type: str = "application/octet-stream") -> None:
        """Upload raw bytes to S3."""
        self._client.put_object(
            Bucket=self.bucket,
            Key=s3_key,
            Body=data,
            ContentType=content_type,
        )

    def download_bytes(self, s3_key: str) -> bytes:
        """Download an S3 object as raw bytes."""
        response = self._client.get_object(Bucket=self.bucket, Key=s3_key)
        return response["Body"].read()

    def append_jsonl(self, record: Any, s3_key: str) -> None:
        """Append a JSON record as a new line to an S3 object.

        Downloads existing content (if any), appends the new line, re-uploads.
        Not atomic — suitable for low-frequency feedback writes.
        """
        try:
            existing = self.download_bytes(s3_key).decode("utf-8")
        except ClientError as e:
            if e.response["Error"]["Code"] == "NoSuchKey":
                existing = ""
            else:
                raise

        new_line = json.dumps(record, default=str) + "\n"
        updated = existing + new_line
        self.upload_bytes(updated.encode("utf-8"), s3_key, content_type="application/x-ndjson")

    def object_exists(self, s3_key: str) -> bool:
        """Return True if the S3 object exists."""
        try:
            self._client.head_object(Bucket=self.bucket, Key=s3_key)
            return True
        except ClientError:
            return False

    def list_keys(self, prefix: str) -> List[str]:
        """List all S3 object keys under a given prefix (recursive)."""
        paginator = self._client.get_paginator("list_objects_v2")
        keys: List[str] = []
        for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                keys.append(obj["Key"])
        return keys

    # ------------------------------------------------------------------
    # Convenience path builders
    # ------------------------------------------------------------------

    def results_key(self, run_name: str, filename: str) -> str:
        """Build S3 key for a benchmark run output file."""
        return f"{self.prefix_results}/{run_name}/{filename}"

    def dataset_key(self, dataset_name: str, relative_path: str) -> str:
        """Build S3 key for a dataset file."""
        return f"{self.prefix_datasets}/{dataset_name}/{relative_path}"

    def pdfs_key(self, dataset_name: str, filename: str) -> str:
        """Build S3 key for a raw PDF under the dataset prefix.

        PDFs are stored at: benchmarking/datasets/{dataset}/pdfs/{filename}
        Upload PDFs there before running parse_documents_aws.py.
        """
        return f"{self.prefix_datasets}/{dataset_name}/pdfs/{filename}"

    def image_key(self, dataset_name: str, relative_path: str) -> str:
        """Build S3 key for an image file under the images prefix.

        `relative_path` should include the filename and extension, e.g.:
            "figures/doc_p01_fig_00.png"
            "figures/doc_p01_page.png"
        """
        return f"{self.prefix_images}/{dataset_name}/{relative_path}"

    def feedback_key(self) -> str:
        """Build S3 key for the feedback JSONL file."""
        return f"{self.prefix_feedback}/feedback.jsonl"

    def query_upload_key(self, filename: str) -> str:
        """Build S3 key for an uploaded query image."""
        return f"{self.prefix_images}/query_uploads/{filename}"
