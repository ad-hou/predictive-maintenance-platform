"""Optional S3 transport for the champion model (used on EC2: the API pulls the model at start)."""

from __future__ import annotations

from pathlib import Path


def parse_uri(uri: str) -> tuple[str, str]:
    if not uri.startswith("s3://"):
        raise ValueError(f"not an s3 uri: {uri}")
    bucket, _, key = uri[5:].partition("/")
    if not bucket or not key:
        raise ValueError(f"s3 uri needs a bucket and a key: {uri}")
    return bucket, key


def download(uri: str, destination: Path, client=None) -> bool:
    """Download uri to destination. Returns False (and keeps any local file) if it does not exist."""
    bucket, key = parse_uri(uri)
    if client is None:
        import boto3

        client = boto3.client("s3")
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        client.download_file(bucket, key, str(destination))
    except Exception:  # missing object or no permission: the API reports "degraded" on /health
        return False
    return True


def upload(path: Path, uri: str, client=None) -> None:
    bucket, key = parse_uri(uri)
    if client is None:
        import boto3

        client = boto3.client("s3")
    client.upload_file(str(path), bucket, key)
