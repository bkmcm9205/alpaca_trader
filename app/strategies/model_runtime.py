# app/strategies/model_runtime.py
from __future__ import annotations
import os, io, json, pickle
from dataclasses import dataclass
from typing import Optional, Tuple, Dict
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path

import boto3
import pandas as pd
import numpy as np

ET = ZoneInfo("America/New_York")

# ---------------- ENV ----------------
S3_BUCKET       = os.getenv("S3_BUCKET", "")
S3_REGION       = os.getenv("S3_REGION", "us-east-2")
S3_BASE_PREFIX  = os.getenv("S3_BASE_PREFIX", "models").strip("/")

MODEL_RELOAD_WINDOW_ET = os.getenv("MODEL_RELOAD_WINDOW_ET", "08:00-09:25")

# Hard switches for troubleshooting
DISABLE_S3           = os.getenv("DISABLE_S3", "0").lower() in ("1","true","yes")
FORCE_MODEL_RELOAD   = os.getenv("FORCE_MODEL_RELOAD", "0").lower() in ("1","true","yes")
VERBOSE_MODEL_LOGS   = os.getenv("VERBOSE_MODEL_LOGS", "1").lower() in ("1","true","yes")

def _override_env_key(strat: str) -> str:
    return f"MODEL_KEY_OVERRIDE_{strat.upper()}"

# ---------------- IO helpers ----------------
def _s3():
    return boto3.client(
        "s3",
        region_name=S3_REGION,
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    )

def _in_window(win: str) -> bool:
    if FORCE_MODEL_RELOAD:
        return True
    try:
        s, e = win.split("-", 1)
        now = datetime.now(ET).strftime("%H:%M")
        return s <= now <= e
    except Exception:
        return True

def _log(msg: str):
    if VERBOSE_MODEL_LOGS:
        print(msg, flush=True)

def _deserialize(art_bytes: bytes) -> dict:
    # Try pickle first
    try:
        return pickle.load(io.BytesIO(art_bytes))
    except Exception:
        pass
    # Try JSON
    try:
        j = json.loads(art_bytes.decode("utf-8"))
        if isinstance(j, dict):
            return {
                "clf": None,
                "prior": float(j.get("prior", 0.5)),
                "features": j.get("features", ["return","rsi","volatility"]),
                "raw": j,
            }
    except Exception:
        pass
    return {"clf": None, "prior": 0.5, "features": ["return","rsi","volatility"]}

def _read_local(path: str) -> Optional[bytes]:
    p = Path(path)
    if p.is_file():
        try:
            return p.read_bytes()
        except Exception:
            return None
    return None

def _read_s3_obj(bucket: str, key: str) -> Optional[bytes]:
    try:
        obj = _s3().get_object(Bucket=bucket, Key=key)
        return obj["Body"].read()
    except Exception:
        return None

def _read_sidecar_meta(bucket: str, key: str) -> Optional[dict]:
    for mk in (key.replace("artifact.bin","meta.json"),
               key.replace("model.pkl","meta.json")):
        b = _read_s3_obj(bucket, mk)
        if b:
            try:
                return json.loads(b.decode("utf-8"))
            except Exception:
                return None
    return None

# ---------------- Cache ----------------
@dataclass
class ModelCache:
    obj: Optional[object] = None
    loaded_day: Optional[datetime.date] = None
    source: str = ""
    meta: Dict = None

_model_caches: Dict[str, ModelCache] = {}

# ---------------- Resolver ----------------
def _resolve_override_ref(strat: str) -> Optional[tuple[str, str]]:
    envk = _override_env_key(strat)
    ref = os.getenv(envk, "").strip()
    if not ref:
        return None
    # local path
    if ref.startswith("/") or ref.startswith("app/") or ref.startswith("./"):
        return ("local", ref)
    # s3 uri
    if ref.startswith("s3://"):
        rest = ref[5:]
        if "/" in rest:
            bucket, key = rest.split("/", 1)
            return ("s3", f"{bucket}/{key}")
        return None
    # bucket-relative key
    if "/" in ref:
        if not S3_BUCKET:
            return None
        return ("s3", f"{S3_BUCKET}/{ref}")
    return None

def _candidate_s3_keys(strat: str) -> list[str]:
    p = S3_BASE_PREFIX
    return [
        f"{p}/{strat}/production/artifact.bin",
        f"{p}/{strat}/production/model.pkl",
        f"{p}/{strat}/prod/artifact.bin",
        f"{p}/{strat}/prod/model.pkl",
        f"{p}/{strat}/artifact.bin",
        f"{p}/{strat}/model.pkl",
    ]

# ---------------- Public API ----------------
def maybe_load_model(strategy_name: str) -> Optional[object]:
    today = datetime.now(ET).date()
    cache = _model_caches.get(strategy_name)
    if cache and cache.obj is not None and cache.loaded_day == today and not FORCE_MODEL_RELOAD:
        return cache.obj

    if not _in_window(MODEL_RELOAD_WINDOW_ET):
        return cache.obj if cache else None

    # 1) Hard override first (no guessing)
    ovr = _resolve_override_ref(strategy_name)
    if ovr:
        kind, ref = ovr
        if kind == "local":
            data = _read_local(ref)
            if data:
                obj = _deserialize(data)
                _model_caches[strategy_name] = ModelCache(obj=obj, loaded_day=today, source=f"local:{ref}", meta={})
                _log(f"[MODEL:{strategy_name}] loaded OVERRIDE local:{ref}")
                return obj
            _log(f"[MODEL:{strategy_name}] OVERRIDE local not found: {ref}")
        else:
            bucket, key = ref.split("/", 1)
            data = _read_s3_obj(bucket, key)
            if data:
                obj = _deserialize(data)
                meta = _read_sidecar_meta(bucket, key) or {}
                _model_caches[strategy_name] = ModelCache(obj=obj, loaded_day=today, source=f"s3://{bucket}/{key}", meta=meta)
                _log(f"[MODEL:{strategy_name}] loaded OVERRIDE s3://{bucket}/{key} metric={meta.get('metric')}")
                return obj
            _log(f"[MODEL:{strategy_name}] OVERRIDE s3 not found: s3://{bucket}/{key}")

    # 2) If override not set or failed: try local repo default spots
    local_candidates = [
        f"app/strategies/models/{strategy_name}/production/model.pkl",
        f"app/strategies/models/{strategy_name}/production/artifact.bin",
        f"app/strategies/models/{strategy_name}/prod/model.pkl",
        f"app/strategies/models/{strategy_name}/prod/artifact.bin",
    ]
    for lp in local_candidates:
        data = _read_local(lp)
        if data:
            obj = _deserialize(data)
            _model_caches[strategy_name] = ModelCache(obj=obj, loaded_day=today, source=f"local:{lp}", meta={})
            _log(f"[MODEL:{strategy_name}] loaded local:{lp}")
            return obj

    # 3) S3 (unless disabled)
    if not DISABLE_S3 and S3_BUCKET:
        for key in _candidate_s3_keys(strategy_name):
            data = _read_s3_obj(S3_BUCKET, key)
            if data is None:
                continue
            obj = _deserialize(data)
            meta = _read_sidecar_meta(S3_BUCKET, key) or {}
            _model_caches[strategy_name] = ModelCache(obj=obj, loaded_day=today, source=f"s3://{S3_BUCKET}/{key}", meta=meta)
            _log(f"[MODEL:{strategy_name}] loaded s3://{S3_BUCKET}/{key} metric={meta.get('metric')}")
            return obj

    _log(f"[MODEL:{strategy_name}] no model found (override/local/S3).")
    return None

# ---------------- Feature pipeline ----------------
def resample_tf(df1m: pd.DataFrame, tf_min: int) -> pd.DataFrame:
    if df1m is None or df1m.empty or "close" not in df1m.columns:
        return pd.DataFrame()
    if tf_min <= 1:
        return df1m.copy()
    rule = f"{int(tf_min)}min"
    agg = {"open":"first","high":"max","low":"min","close":"last","volume":"sum"}
    try:
        bars = df1m.resample(rule, origin="start_day", label="right").agg(agg).dropna()
    except Exception:
        bars = pd.DataFrame()
    return bars

def latest_features(bars: pd.DataFrame) -> Tuple[Optional[pd.DataFrame], Optional[float]]:
    if bars is None or bars.empty or "close" not in bars.columns:
        return None, None
    df = pd.DataFrame(index=bars.index)
    df["close"] = bars["close"]
    df["return"] = df["close"].pct_change()
    delta = df["close"].diff()
    up = delta.clip(lower=0).rolling(14).mean()
    down = -delta.clip(upper=0).rolling(14).mean()
    rs = up / (down.replace(0, np.nan))
    df["rsi"] = 100 - (100 / (1 + rs))
    df["volatility"] = df["return"].rolling(20).std()
    df = df.dropna()
    if df.empty:
        return None, None
    x = df[["return","rsi","volatility"]].iloc[[-1]]
    last = float(df["close"].iloc[-1])
    return x, last

def proba_up_for(model_obj: object, x_live: pd.DataFrame) -> Optional[float]:
    if model_obj is None:
        return None
    try:
        clf = model_obj.get("clf")
        if clf is None:
            return float(model_obj.get("prior", 0.5))
        feats = model_obj.get("features", ["return","rsi","volatility"])
        x = x_live
        try:
            x = x_live[feats]
        except Exception:
            pass
        return float(clf.predict_proba(x)[0][1])
    except Exception:
        return None
