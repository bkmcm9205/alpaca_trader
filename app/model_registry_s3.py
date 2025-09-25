from __future__ import annotations
import os, io, json, pickle, time, uuid, logging
import boto3
from typing import Any, Optional

log = logging.getLogger("model_registry_s3")

class S3ModelRegistry:
    """
    Layout:
      s3://{bucket}/{base}/<strategy>/production/model.pkl
      s3://{bucket}/{base}/<strategy>/production/metrics.json
      s3://{bucket}/{base}/<strategy>/candidates/<timestamp>/model.pkl
      s3://{bucket}/{base}/<strategy>/candidates/<timestamp>/metrics.json
    """
    def __init__(self, bucket: str, region: str, base_prefix: str):
        self.bucket = bucket
        self.base = base_prefix.strip("/").rstrip("/")
        self.s3 = boto3.client("s3", region_name=region) if region else boto3.client("s3")

    def _key(self, *parts) -> str:
        return "/".join([p.strip("/") for p in (self.base,)+parts])

    def load_production(self, strategy: str) -> Any:
        key = self._key(strategy, "production", "model.pkl")
        log.info(f"[S3] downloading production model: s3://{self.bucket}/{key}")
        obj = self.s3.get_object(Bucket=self.bucket, Key=key)
        return pickle.loads(obj["Body"].read())

    def load_production_metrics(self, strategy: str) -> Optional[dict]:
        key = self._key(strategy, "production", "metrics.json")
        try:
            obj = self.s3.get_object(Bucket=self.bucket, Key=key)
            return json.loads(obj["Body"].read().decode())
        except self.s3.exceptions.NoSuchKey:
            return None
        except Exception:
            return None

    def save_candidate(self, strategy: str, model_obj: Any, metrics: dict) -> str:
        ts = int(time.time())
        cand_prefix = self._key(strategy, "candidates", str(ts))
        mpkl = pickle.dumps(model_obj)
        self.s3.put_object(Bucket=self.bucket, Key=f"{cand_prefix}/model.pkl", Body=mpkl)
        self.s3.put_object(Bucket=self.bucket, Key=f"{cand_prefix}/metrics.json",
                           Body=json.dumps(metrics).encode(), ContentType="application/json")
        log.info(f"[S3] saved candidate at s3://{self.bucket}/{cand_prefix}")
        return f"{cand_prefix}"

    def promote(self, strategy: str, candidate_prefix: str):
        # copy candidate -> production
        src_model = f"{candidate_prefix}/model.pkl"
        src_metrics = f"{candidate_prefix}/metrics.json"
        dst_model = self._key(strategy, "production", "model.pkl")
        dst_metrics = self._key(strategy, "production", "metrics.json")

        self.s3.copy_object(Bucket=self.bucket, CopySource={"Bucket": self.bucket, "Key": src_model}, Key=dst_model)
        self.s3.copy_object(Bucket=self.bucket, CopySource={"Bucket": self.bucket, "Key": src_metrics}, Key=dst_metrics)
        log.info(f"[S3] promoted candidate -> production for {strategy}")

    def maybe_promote_if_better(self, strategy: str, candidate_prefix: str, metric_name: str="val_sharpe"):
        # Compare one scalar metric (higher is better)
        prod = self.load_production_metrics(strategy) or {}
        try:
            prod_val = float(prod.get(metric_name, float("-inf")))
        except Exception:
            prod_val = float("-inf")

        obj = self.s3.get_object(Bucket=self.bucket, Key=f"{candidate_prefix}/metrics.json")
        cand = json.loads(obj["Body"].read().decode())
        cand_val = float(cand.get(metric_name, float("-inf")))

        if cand_val > prod_val:
            self.promote(strategy, candidate_prefix)
            return True, cand_val, prod_val
        return False, cand_val, prod_val
