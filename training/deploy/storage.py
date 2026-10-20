"""S3-compatible object storage adapter for the training deployment."""

from __future__ import annotations

import boto3
import os
from botocore.exceptions import ClientError
from pathlib import Path


class ObjectStorage:
    def __init__(
        self,
        bucket=None,
        endpoint_url=None,
        region=None,
        access_key=None,
        secret_key=None,
    ):
        self.bucket = bucket or os.getenv("OBJECT_STORAGE_BUCKET", "")
        if not self.bucket:
            raise ValueError("OBJECT_STORAGE_BUCKET must be configured")
        self.client = boto3.client(
            "s3",
            endpoint_url=endpoint_url or os.getenv("S3_ENDPOINT_URL") or None,
            region_name=region or os.getenv("AWS_REGION", "us-east-1"),
            aws_access_key_id=access_key or os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=secret_key or os.getenv("AWS_SECRET_ACCESS_KEY"),
        )

    def ensure_bucket_access(self):
        self.client.head_bucket(Bucket=self.bucket)

    def list_keys(self, prefix: str):
        paginator = self.client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
            for item in page.get("Contents", []):
                yield item["Key"]

    def download(self, key: str, destination: str | Path):
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        self.client.download_file(self.bucket, key, str(destination))
        return destination

    def upload(
        self, source: str | Path, key: str, content_type="application/octet-stream"
    ):
        self.client.upload_file(
            str(source), self.bucket, key, ExtraArgs={"ContentType": content_type}
        )
        return key

    def upload_json(self, key: str, value):
        import json

        self.client.put_object(
            Bucket=self.bucket,
            Key=key,
            Body=json.dumps(value, indent=2, default=str).encode("utf-8"),
            ContentType="application/json",
        )
        return key

    def read_json(self, key: str):
        import json

        obj = self.client.get_object(Bucket=self.bucket, Key=key)
        return json.loads(obj["Body"].read().decode("utf-8"))
