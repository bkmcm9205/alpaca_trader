# app/strategies/model_runtime.py
from __future__ import annotations
import os, pickle, io, json
from dataclasses import dataclass
from typing import Optional, Tuple, Dict
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
import numpy as np

from app.model_registry_s3 import S3ModelRegistry

ET = ZoneInfo("America/New_York")

# ---------------------------------
# S3-based pre-market model loader
# ---------------------------------

@dataclass
class ModelCache:
    obj: Optional[object] = None
    loaded_day: Optional[datetime.date] = None
    vtag: str = ""
    meta: Dict = None

_model_caches: Dict[str, ModelCache] = {}

# one registry for all strategies
_registry = S3ModelRegistry(
    bucket=os.getenv("S3_BUCKET", ""),
    region=os.getenv("S3_REGION", "us-east-2"),
    base_prefix=os.getenv("S3_BASE_PREFIX", "models"),
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

def _deserialize_artifact(art_bytes: bytes) -> dict:
    """
    Accept either:
      - pickle bytes of a dict like {"clf": sklearn_model, "features":[...], ...}
      - JSON bytes of a dict (older trainer artifact). We wrap it into a trivial prior model.
    Returns a dict that proba_up_for() can handle.
    """
    # Try pickle first
    try:
        return pickle.load(io.BytesIO(art_bytes))
    except Exception:
        pass

    # Try JSON next
    try:
        j = json.loads(art_bytes.decode("utf-8"))
        if isinstance(j, dict):
            # Normalize to a trivial prior model; keep common defaults
            return {
                "clf": None,
                "prior": float(j.get("prior", 0.5)),
                "features": j.get("features", ["return", "rsi", "volatility"]),
                "raw": j,
            }
    except Exception:
        pass

    # Last resort: return a safe neutral prior so the strategy can still run
    return {"clf": None, "prior": 0.5, "features": ["return", "rsi", "volatility"]}

def maybe_load_model(strategy_name: str) -> Optional[object]:
    """
    Load models/<strategy>/production/artifact.bin **only during MODEL_RELOAD_WINDOW_ET**,
    at most once per ET day. Returns the deserialized object or None.
    """
    window = os.getenv("MODEL_RELOAD_WINDOW_ET", "08:00-09:25")
    if not _in_window(window):
        # outside reload window → keep current in-memory copy
        return _model_caches.get(strategy_name, ModelCache()).obj

    today = datetime.now(ET).date()
    cache = _model_caches.get(strategy_name)
    if cache and cache.obj is not None and cache.loaded_day == today:
        return cache.obj

    # try loading from registry
    try:
        art, meta, vtag = _registry.load_production(strategy_name)
        model_obj = _deserialize_artifact(art)
        _model_caches[strategy_name] = ModelCache(
            obj=model_obj, loaded_day=today, vtag=vtag, meta=meta
        )
        metric = meta.get("metric")
        print(
            f"[MODEL:{strategy_name}] loaded from S3 metric={metric} vtag={vtag[:8]}…",
            flush=True,
        )
        return model_obj
    except FileNotFoundError as e:
        print(f"[MODEL:{strategy_name}] no production artifact in S3 → {e}", flush=True)
        return None
    except Exception as e:
        print(f"[MODEL:{strategy_name}] failed to load/deserialize → {e}", flush=True)
        return None

# ---------------------------------
# Feature pipeline (match trainer)
# ---------------------------------

def resample_tf(df1m: pd.DataFrame, tf_min: int) -> pd.DataFrame:
    if df1m is None or df1m.empty or "close" not in df1m.columns:
        return pd.DataFrame()
    if tf_min <= 1:
        return df1m.copy()
    rule = f"{int(tf_min)}min"
    agg = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    try:
        bars = df1m.resample(rule, origin="start_day", label="right").agg(agg).dropna()
    except Exception:
        bars = pd.DataFrame()
    return bars

def latest_features(bars: pd.DataFrame) -> Tuple[Optional[pd.DataFrame], Optional[float]]:
    """
    Returns (X_live, last_price) where X_live is a 1-row DataFrame
    with columns [return, rsi, volatility] aligned to trainer.
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

    x = df[["return", "rsi", "volatility"]].iloc[[-1]]
    last = float(df["close"].iloc[-1])
    return x, last

# ---------------------------------
# Inference wrapper
# ---------------------------------

def proba_up_for(model_obj: object, x_live: pd.DataFrame) -> Optional[float]:
    """
    Works with:
      - {"clf": sklearn_model, "features":[...]}
      - {"clf": None, "prior": 0.5}  (JSON/placeholder artifacts)
    """
    if model_obj is None:
        return None
    try:
        clf = model_obj.get("clf")
        if clf is None:
            return float(model_obj.get("prior", 0.5))
        feats = model_obj.get("features", ["return", "rsi", "volatility"])
        x = x_live
        try:
            x = x_live[feats]
        except Exception:
            pass
        return float(clf.predict_proba(x)[0][1])
    except Exception:
        return None
