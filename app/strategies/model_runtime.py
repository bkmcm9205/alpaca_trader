# app/strategies/model_runtime.py
from __future__ import annotations
import os, pickle
from dataclasses import dataclass
from typing import Optional, Tuple, Dict
from datetime import datetime
from zoneinfo import ZoneInfo

import boto3
import pandas as pd
import numpy as np

ET = ZoneInfo("America/New_York")

# -----------------------------
# S3 + Pre-market model loader
# -----------------------------

@dataclass
class ModelCache:
    obj: Optional[object] = None
    loaded_day: Optional[datetime.date] = None
    path: str = ""

_model_caches: Dict[str, ModelCache] = {}

def _s3():
    return boto3.client(
        "s3",
        region_name=os.getenv("S3_REGION", "us-east-2"),
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    )

def _in_window(window: str) -> bool:
    """
    Inclusive HH:MM-HH:MM in ET. If parse fails, be permissive (True).
    """
    try:
        s, e = window.split("-", 1)
        now = datetime.now(ET).strftime("%H:%M")
        return s <= now <= e
    except Exception:
        return True

def _model_dir_for(strategy_name: str) -> str:
    """
    Reads per-strategy env like MODEL_DIR_TRADING_BOT, fallback to models/<strat>/prod
    """
    key = f"MODEL_DIR_{strategy_name.upper()}"
    default = f"models/{strategy_name}/prod"
    return os.getenv(key, default)

def maybe_load_model(strategy_name: str) -> Optional[object]:
    """
    Load models/<strategy>/prod/model.pkl ONLY during MODEL_RELOAD_WINDOW_ET,
    at most once per ET day. Returns the unpickled object or None.
    """
    global _model_caches
    window = os.getenv("MODEL_RELOAD_WINDOW_ET", "08:00-09:25")
    if not _in_window(window):
        # Outside the reload window — keep current model in memory
        return _model_caches.get(strategy_name, ModelCache()).obj

    today = datetime.now(ET).date()
    cache = _model_caches.get(strategy_name)
    if cache and cache.obj is not None and cache.loaded_day == today:
        return cache.obj

    bucket = os.getenv("S3_BUCKET", "")
    if not bucket:
        return cache.obj if cache else None

    model_dir = _model_dir_for(strategy_name)
    key = f"{model_dir.rstrip('/')}/model.pkl"

    try:
        body = _s3().get_object(Bucket=bucket, Key=key)["Body"].read()
        obj = pickle.loads(body)
        _model_caches[strategy_name] = ModelCache(obj=obj, loaded_day=today, path=f"s3://{bucket}/{key}")
        print(f"[MODEL:{strategy_name}] loaded {_model_caches[strategy_name].path}", flush=True)
        return obj
    except Exception as e:
        print(f"[MODEL:{strategy_name}] load failed ({key}): {e}", flush=True)
        # Keep whatever we had
        return cache.obj if cache else None

# -----------------------------
# Feature pipeline (match trainer)
# -----------------------------

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
    """
    Returns (X_live, last_price) where X_live is a 1-row DataFrame
    with columns ["return","rsi","volatility"] aligned to trainer.
    """
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

# -----------------------------
# Inference wrapper
# -----------------------------

def proba_up_for(model_obj: object, x_live: pd.DataFrame) -> Optional[float]:
    """
    Works with the trainer’s artifact: either {"clf": RF, "features":[...]}
    or a trivial {"clf": None, "prior": 0.5}.
    """
    if model_obj is None:
        return None
    try:
        clf = model_obj.get("clf")
        if clf is None:
            # trivial prior model
            return float(model_obj.get("prior", 0.5))
        # align features if provided
        feats = model_obj.get("features", ["return","rsi","volatility"])
        x = x_live
        try:
            x = x_live[feats]
        except Exception:
            pass
        p = float(clf.predict_proba(x)[0][1])
        return p
    except Exception:
        return None
