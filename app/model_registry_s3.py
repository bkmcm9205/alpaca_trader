from __future__ import annotations
import os, io, json, time, hashlib
from typing import Optional, Tuple
import boto3
from botocore.exceptions import ClientError

class S3ModelRegistry:
    """
    s3://<bucket>/<base>/models/<strategy>/
      production/{artifact.bin, meta.json}
      candidates/<ts>/{artifact.bin, meta.json}
    """
    def __init__(self, bucket: str, region: str, base_prefix: str = "models"):
        if not bucket or not region:
            raise ValueError("S3ModelRegistry requires S3 bucket and region")
        self.bucket = bucket
        self.region = region
        self.base = base_prefix.strip("/")
        self.s3 = boto3.client(
            "s3",
            region_name=region,
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        )

    def _pfx(self, strategy: str) -> str:
        return f"{self.base}/{strategy}"

    def _prod_keys(self, strategy: str):
        p = self._pfx(strategy)
        return f"{p}/production/artifact.bin", f"{p}/production/meta.json"

    def _cand_keys(self, strategy: str, ts: str):
        p = self._pfx(strategy)
        return f"{p}/candidates/{ts}/artifact.bin", f"{p}/candidates/{ts}/meta.json"

    def load_production(self, strategy: str):
        art_key, meta_key = self._prod_keys(strategy)
        try:
            meta_obj = self.s3.get_object(Bucket=self.bucket, Key=meta_key)
            art_obj = self.s3.get_object(Bucket=self.bucket, Key=art_key)
        except ClientError as e:
            raise FileNotFoundError(f"No production model for {strategy}: {e}")
        meta = json.loads(meta_obj["Body"].read().decode("utf-8"))
        art = art_obj["Body"].read()
        vtag = hashlib.sha256(art).hexdigest()
        return art, meta, vtag

    def save_candidate(self, strategy: str, artifact: bytes, meta: dict) -> str:
        ts = meta.get("ts") or time.strftime("%Y%m%d-%H%M%S", time.gmtime())
        art_key, meta_key = self._cand_keys(strategy, ts)
        self.s3.put_object(Bucket=self.bucket, Key=art_key, Body=artifact)
        self.s3.put_object(Bucket=self.bucket, Key=meta_key, Body=json.dumps(meta).encode("utf-8"))
        return ts

    def promote_candidate(self, strategy: str, ts: str):
        src_art, src_meta = self._cand_keys(strategy, ts)
        dst_art, dst_meta = self._prod_keys(strategy)
        self.s3.copy_object(Bucket=self.bucket, CopySource={"Bucket": self.bucket, "Key": src_art}, Key=dst_art)
        self.s3.copy_object(Bucket=self.bucket, CopySource={"Bucket": self.bucket, "Key": src_meta}, Key=dst_meta)

    def _get_candidate(self, strategy: str, ts: str):
        art_key, meta_key = self._cand_keys(strategy, ts)
        meta_obj = self.s3.get_object(Bucket=self.bucket, Key=meta_key)
        art_obj = self.s3.get_object(Bucket=self.bucket, Key=art_key)
        return art_obj["Body"].read(), json.loads(meta_obj["Body"].read().decode("utf-8"))

    def compare_and_maybe_promote(self, strategy: str, candidate_ts: str, higher_is_better: bool = True) -> bool:
        _, cmeta = self._get_candidate(strategy, candidate_ts)
        c = float(cmeta.get("metric", float("nan")))
        try:
            _, pmeta, _ = self.load_production(strategy)
            p = float(pmeta.get("metric", float("nan")))
            better = (c > p) if higher_is_better else (c < p)
        except FileNotFoundError:
            better = True
        if better:
            self.promote_candidate(strategy, candidate_ts)
        return better
