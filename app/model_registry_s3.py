from __future__ import annotations
import os, io, json, time, hashlib
from typing import Optional, Tuple
import boto3
from botocore.exceptions import ClientError

class S3ModelRegistry:
    """
    Layout:
      s3://<bucket>/<base>/models/<strategy>/
        production/
          artifact.bin      (your model bytes: pickle, JSON, whatever)
          meta.json         ({"metric": float, "ts": "...", ...})
        candidates/<ts>/
          artifact.bin
          meta.json
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

    def _prod_keys(self, strategy: str) -> Tuple[str, str]:
        pfx = self._pfx(strategy)
        return f"{pfx}/production/artifact.bin", f"{pfx}/production/meta.json"

    def _cand_keys(self, strategy: str, ts: str) -> Tuple[str, str]:
        pfx = self._pfx(strategy)
        return f"{pfx}/candidates/{ts}/artifact.bin", f"{pfx}/candidates/{ts}/meta.json"

    # ------------ LOAD ------------
    def load_production(self, strategy: str) -> Tuple[bytes, dict, str]:
        """Returns (artifact_bytes, meta_dict, version_tag)."""
        art_key, meta_key = self._prod_keys(strategy)
        try:
            meta_obj = self.s3.get_object(Bucket=self.bucket, Key=meta_key)
            art_obj = self.s3.get_object(Bucket=self.bucket, Key=art_key)
        except ClientError as e:
            raise FileNotFoundError(f"No production model for {strategy}: {e}")

        meta = json.loads(meta_obj["Body"].read().decode("utf-8"))
        art_bytes = art_obj["Body"].read()
        # version tag = sha256 of artifact for quick change detection
        vtag = hashlib.sha256(art_bytes).hexdigest()
        return art_bytes, meta, vtag

    # ------------ SAVE CANDIDATE ------------
    def save_candidate(self, strategy: str, artifact: bytes, meta: dict) -> str:
        ts = meta.get("ts") or time.strftime("%Y%m%d-%H%M%S", time.gmtime())
        art_key, meta_key = self._cand_keys(strategy, ts)
        self.s3.put_object(Bucket=self.bucket, Key=art_key, Body=artifact)
        self.s3.put_object(Bucket=self.bucket, Key=meta_key, Body=json.dumps(meta).encode("utf-8"))
        return ts  # candidate id

    # ------------ PROMOTE ------------
    def promote_candidate(self, strategy: str, candidate_ts: str):
        src_art, src_meta = self._cand_keys(strategy, candidate_ts)
        dst_art, dst_meta = self._prod_keys(strategy)
        # Copy over candidate -> production
        self.s3.copy_object(Bucket=self.bucket, CopySource={"Bucket": self.bucket, "Key": src_art}, Key=dst_art)
        self.s3.copy_object(Bucket=self.bucket, CopySource={"Bucket": self.bucket, "Key": src_meta}, Key=dst_meta)

    # ------------ HELPER: compare & promote ------------
    def compare_and_maybe_promote(self, strategy: str, candidate_ts: str, higher_is_better: bool = True) -> bool:
        """Compares candidate metric to production; promotes if better or if no production exists."""
        _, cand_meta = self._get_candidate(strategy, candidate_ts)
        c = float(cand_meta.get("metric", float("nan")))

        try:
            _, prod_meta, _ = self.load_production(strategy)
            p = float(prod_meta.get("metric", float("nan")))
            is_better = (c > p) if higher_is_better else (c < p)
        except FileNotFoundError:
            is_better = True  # no production yet

        if is_better:
            self.promote_candidate(strategy, candidate_ts)
        return is_better

    def _get_candidate(self, strategy: str, ts: str) -> Tuple[bytes, dict]:
        art_key, meta_key = self._cand_keys(strategy, ts)
        meta_obj = self.s3.get_object(Bucket=self.bucket, Key=meta_key)
        art_obj = self.s3.get_object(Bucket=self.bucket, Key=art_key)
        return art_obj["Body"].read(), json.loads(meta_obj["Body"].read().decode("utf-8"))
