from __future__ import annotations
import os, time, json, pickle
from datetime import datetime
from zoneinfo import ZoneInfo
import pandas as pd

from app.model_registry_s3 import S3ModelRegistry

# Import your existing code. If your function/class name differs, tweak _call() accordingly.
# Common patterns your file may expose: compute_signal(...) OR signal_ml_pattern(...)
try:
    from app.strategies.trading_bot import compute_signal as _compute  # type: ignore
except Exception:
    from app.strategies.trading_bot import signal_ml_pattern as _compute  # type: ignore

NAME = "trading_bot"

# --------- S3 model loading (read-only) ----------
_S3_BUCKET = os.getenv("S3_BUCKET", "")
_S3_REGION = os.getenv("S3_REGION", "")
_S3_PREFIX = os.getenv("S3_BASE_PREFIX", "models")
_registry = S3ModelRegistry(_S3_BUCKET, _S3_REGION, _S3_PREFIX) if _S3_BUCKET and _S3_REGION else None

# Hot-reload policy: only pre-market, once per day
_MODEL_RELOAD_WINDOW = os.getenv("MODEL_RELOAD_WINDOW_ET", "08:00-09:25")
_REFRESH_SEC = int(os.getenv("MODEL_REFRESH_SEC", "60"))

_model_blob, _model_meta, _vtag, _last_check, _loaded_day = None, {}, "", 0.0, None

def _deserialize(b: bytes):
    try:
        return pickle.loads(b)
    except Exception:
        try:
            return json.loads(b.decode("utf-8"))
        except Exception:
            return b  # raw bytes

def _in_reload_window_et() -> bool:
    try:
        start_s, end_s = _MODEL_RELOAD_WINDOW.split("-", 1)
        now_hm = datetime.now(ZoneInfo("America/New_York")).strftime("%H:%M")
        return start_s <= now_hm <= end_s
    except Exception:
        return False

def _maybe_load(force_boot: bool = False):
    global _model_blob, _model_meta, _vtag, _last_check, _loaded_day
    if not _registry:
        return
    now = time.time()

    if force_boot and not _vtag:
        pass  # allow first load at boot any time
    else:
        if (now - _last_check) < _REFRESH_SEC:
            return
        _last_check = now
        if not _in_reload_window_et():
            return
        today = datetime.now(ZoneInfo("America/New_York")).date()
        if _loaded_day == today:
            return

    try:
        art, meta, vtag = _registry.load_production(NAME)
        if vtag != _vtag:
            _model_blob, _model_meta, _vtag = _deserialize(art), meta, vtag
            _loaded_day = datetime.now(ZoneInfo("America/New_York")).date()
            print(f"[{NAME}] loaded production model vtag={_vtag[:8]} metric={meta.get('metric')}", flush=True)
        else:
            _loaded_day = datetime.now(ZoneInfo("America/New_York")).date()
    except Exception:
        # ok if no production yet
        pass

def _call(symbol: str, df1m: pd.DataFrame, tf_minutes: int):
    # Try most common signatures your code uses
    try:
        return _compute("ml_pattern", symbol, tf_minutes, df1m=df1m, model=_model_blob)
    except TypeError:
        try:
            return _compute("ml_pattern", symbol, tf_minutes, df1m=df1m)
        except TypeError:
            return _compute(symbol, df1m, tf_minutes)

def decide(symbol: str, df1m: pd.DataFrame, tf_minutes: int) -> dict | None:
    _maybe_load(force_boot=True)
    sig = _call(symbol, df1m, tf_minutes)
    if not sig:
        return None
    action = (sig.get("action") or "").lower()
    side = "sell" if action in ("sell", "sell_short", "short") else "buy"
    return {
        "side": side,
        "confidence": float(sig.get("confidence", sig.get("score", 0.0) or 0.0)),
        "quantity": int(sig.get("quantity", 0) or 0),
        "takeProfit": sig.get("takeProfit"),
        "stopLoss": sig.get("stopLoss"),
        "meta": {"src": NAME, **(sig.get("meta") or {}), **({"model_metric": _model_meta.get("metric")} if _model_meta else {})},
    }
