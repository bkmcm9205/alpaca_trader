# app/trainer_worker.py
from __future__ import annotations

import os
import time
import json
import math
from typing import List, Dict, Tuple
from datetime import datetime
from zoneinfo import ZoneInfo

import requests
import pandas as pd
import numpy as np
from botocore.exceptions import ClientError

from app.data_alpaca import fetch_alpaca_1m
from app.model_registry_s3 import S3ModelRegistry

# =========================================================
# ENV & CONSTANTS
# =========================================================

ALPACA_TRADING_BASE = os.getenv("ALPACA_TRADING_BASE", "https://paper-api.alpaca.markets").rstrip("/")
ALPACA_KEY          = os.getenv("ALPACA_KEY_ID", os.getenv("APCA_API_KEY_ID", ""))
ALPACA_SECRET       = os.getenv("ALPACA_SECRET_KEY", os.getenv("APCA_API_SECRET_KEY", ""))

S3_BUCKET      = os.getenv("S3_BUCKET", "")
S3_REGION      = os.getenv("S3_REGION", "")
S3_BASE_PREFIX = os.getenv("S3_BASE_PREFIX", "models")

# Train ALL FOUR by default
TRAIN_STRATEGIES = [s.strip() for s in os.getenv(
    "TRAIN_STRATEGIES",
    "trading_bot,ml_merged,ranked_ml,ml_sentiment"
).split(",") if s.strip()]

# Timeframes to aggregate for features
TRAIN_TFS        = [int(x) for x in os.getenv("TRAIN_TFS", "5,15").split(",") if x.strip()]

# Nightly window
RUN_START_ET = os.getenv("TRAIN_RUN_START_ET", "19:00")
RUN_END_ET   = os.getenv("TRAIN_RUN_END_ET",   "20:00")
TRAINER_SLEEP_SEC = int(os.getenv("TRAINER_SLEEP_SEC", "60"))
TRAIN_FORCE_RUN = os.getenv("TRAIN_FORCE_RUN", "0").lower() in ("1","true","yes")

# Universe
TRAIN_USE_UNIVERSE = os.getenv("TRAIN_USE_UNIVERSE", "1").lower() in ("1","true","yes")
TRAIN_MAX_SYMBOLS  = int(os.getenv("TRAIN_MAX_SYMBOLS", "0"))  # 0 = no cap
UNIVERSE_WRITE_KEY = os.getenv("UNIVERSE_S3_WRITE_KEY", "models/universe/daily.csv")
UNIVERSE_MAX       = int(os.getenv("UNIVERSE_MAX", "8000"))

# Lookback (in minutes) of 1m bars to pull for each symbol (pooled dataset)
LOOKBACK_MIN = int(os.getenv("TRAIN_LOOKBACK_MIN", str(7*24*60)))  # 7 days by default

# Per-strategy quality floors (optional). If not set, default is 0.0 (no floor).
# Set as ENV: PROMOTE_MIN_TRADING_BOT=0.02, etc.
PROMOTE_MIN_DEFAULT = float(os.getenv("PROMOTE_MIN_DEFAULT", "0.0"))

# RF hyperparams (global defaults; can override per strategy via N_ESTIMATORS_<STRAT>, MAX_DEPTH_<STRAT>)
N_ESTIMATORS_DEFAULT = int(os.getenv("N_ESTIMATORS_DEFAULT", "300"))
MAX_DEPTH_DEFAULT    = os.getenv("MAX_DEPTH_DEFAULT", "")  # empty = None

# =========================================================
# UTIL
# =========================================================

def _now_et_hhmm() -> str:
    return datetime.now(ZoneInfo("America/New_York")).strftime("%H:%M")

def _within_window() -> bool:
    now_hm = _now_et_hhmm()
    return RUN_START_ET <= now_hm <= RUN_END_ET

def _auth_headers() -> Dict[str, str]:
    return {"Apca-Api-Key-Id": ALPACA_KEY, "Apca-Api-Secret-Key": ALPACA_SECRET, "accept": "application/json"}

def _per_strat_float(prefix: str, strat: str, default: float) -> float:
    """
    Read a per-strategy float env like PROMOTE_MIN_TRADING_BOT.
    """
    try:
        v = os.getenv(f"{prefix}_{strat.upper()}")
        if v is None:
            return default
        return float(v)
    except Exception:
        return default

def _per_strat_int(prefix: str, strat: str, default: int) -> int:
    v = os.getenv(f"{prefix}_{strat.upper()}")
    try:
        return int(v) if v is not None else default
    except Exception:
        return default

def _per_strat_opt_int(prefix: str, strat: str, default: int | None) -> int | None:
    v = os.getenv(f"{prefix}_{strat.upper()}")
    if v is None or str(v).strip() == "":
        return default
    try:
        return int(v)
    except Exception:
        return default

# =========================================================
# UNIVERSE (assets-only)
# =========================================================

def _list_us_equities() -> List[str]:
    url = f"{ALPACA_TRADING_BASE}/v2/assets"
    params = {"asset_class": "us_equity", "status": "active"}
    r = requests.get(url, headers=_auth_headers(), params=params, timeout=60)
    r.raise_for_status()
    assets = r.json()
    syms = [str(a.get("symbol","")).upper().strip() for a in assets if a.get("tradable")]
    return syms[:UNIVERSE_MAX]

def rebuild_universe_to_s3(registry: S3ModelRegistry):
    if not (S3_BUCKET and S3_REGION and UNIVERSE_WRITE_KEY):
        print("[UNIVERSE] S3 not configured; skipping", flush=True)
        return
    try:
        keep_syms = _list_us_equities()
        print(f"[UNIVERSE] assets-only -> keep {len(keep_syms)} symbols", flush=True)
        csv_bytes = ("\n".join(sorted(set(keep_syms))) + "\n").encode("utf-8")
        registry.s3.put_object(Bucket=S3_BUCKET, Key=UNIVERSE_WRITE_KEY, Body=csv_bytes)
        print(f"[UNIVERSE] wrote {len(set(keep_syms))} symbols to s3://{S3_BUCKET}/{UNIVERSE_WRITE_KEY}", flush=True)
    except Exception as e:
        print(f"[UNIVERSE] assets-only error: {e}", flush=True)

def load_universe_from_s3(registry: S3ModelRegistry) -> List[str]:
    if not (S3_BUCKET and S3_REGION and UNIVERSE_WRITE_KEY):
        return []
    try:
        body = registry.s3.get_object(Bucket=S3_BUCKET, Key=UNIVERSE_WRITE_KEY)["Body"].read().decode("utf-8", errors="ignore")
        out = []
        for ln in body.splitlines():
            s = ln.replace("\ufeff", "").strip().upper()
            if s and not s.startswith("#"):
                out.append(s)
        return out
    except ClientError as e:
        print(f"[UNIVERSE] read error: {e}", flush=True)
        return []

# =========================================================
# SHARED FEATURE PIPELINE
# =========================================================

def _aggregate_bars(df1m: pd.DataFrame, tf_min: int) -> pd.DataFrame:
    if df1m is None or df1m.empty:
        return pd.DataFrame()
    if tf_min <= 1:
        return df1m.copy()
    rule = f"{tf_min}min"
    agg = {
        "open":   "first",
        "high":   "max",
        "low":    "min",
        "close":  "last",
        "volume": "sum",
    }
    try:
        out = df1m.resample(rule).agg(agg).dropna()
    except Exception:
        out = pd.DataFrame()
    return out

def _build_features(bars: pd.DataFrame) -> pd.DataFrame:
    """
    Keep features consistent with what your adapters can compute:
    - return (close pct change)
    - rsi(14) (simple)
    - volatility (20-bar std of return)
    """
    if bars is None or bars.empty or "close" not in bars.columns:
        return pd.DataFrame()
    df = pd.DataFrame(index=bars.index)
    df["close"] = bars["close"]
    df["return"] = df["close"].pct_change()

    delta = df["close"].diff()
    up = delta.clip(lower=0).rolling(14).mean()
    down = -delta.clip(upper=0).rolling(14).mean()
    rs = up / (down.replace(0, np.nan))
    df["rsi"] = 100 - (100 / (1 + rs))
    df["volatility"] = df["return"].rolling(20).std()
    df["target_up"] = (df["close"].shift(-1) > df["close"]).astype(int)
    df = df.dropna()
    if len(df) < 200:
        return pd.DataFrame()
    return df

def _pool_training_data(sym_list: List[str], tfs: List[int], lookback_min: int) -> Tuple[pd.DataFrame, pd.Series]:
    X_parts = []
    y_parts = []
    for sym in sym_list:
        df1m = fetch_alpaca_1m(sym, limit=lookback_min)
        if df1m is None or df1m.empty:
            continue
        for tf in tfs:
            bars = _aggregate_bars(df1m, tf)
            feats = _build_features(bars)
            if feats is None or feats.empty:
                continue
            X_parts.append(feats[["return","rsi","volatility"]].iloc[:-1])  # drop last row (no next label)
            y_parts.append(feats["target_up"].iloc[:-1])
    if not X_parts:
        return pd.DataFrame(), pd.Series(dtype=int)
    X = pd.concat(X_parts, axis=0)
    y = pd.concat(y_parts, axis=0).astype(int)
    return X, y

# =========================================================
# PER-STRATEGY FITTERS (ALL 4 STRATEGIES)
# =========================================================

def _fit_classifier(sym_list: List[str], tfs: List[int], lookback_min: int,
                    n_estimators: int, max_depth: int | None) -> tuple[bytes, dict]:
    """
    Generic RF classifier trainer used by all 4 strategies.
    Returns (artifact_bytes, meta_dict) with artifact = pickle.dumps({"clf": clf, "features": [...]})
    """
    import pickle
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import roc_auc_score, accuracy_score

    X, y = _pool_training_data(sym_list, tfs, lookback_min)
    if X.empty or len(X) < 1000:
        # If dataset is too small, return a trivial model that always predicts prior
        prior = float(y.mean() if len(y) else 0.5)
        trivial = {"clf": None, "features": ["return","rsi","volatility"], "prior": prior}
        meta = {"ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "metric": 0.0, "metric_name":"roc_auc", "accuracy": 0.5,
                "n_samples": int(len(X))}
        return pickle.dumps(trivial), meta

    cut = int(len(X) * 0.7)
    Xtr, ytr = X.iloc[:cut], y.iloc[:cut]
    Xva, yva = X.iloc[cut:], y.iloc[cut:]
    if len(Xva) < 200:
        cut = max(200, int(len(X)*0.8))
        Xtr, ytr = X.iloc[:cut], y.iloc[:cut]
        Xva, yva = X.iloc[cut:], y.iloc[cut:]

    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42,
        n_jobs=-1
    )
    clf.fit(Xtr, ytr)

    yhatp = clf.predict_proba(Xva)[:,1]
    yhat  = (yhatp >= 0.5).astype(int)
    try:
        auc = float(roc_auc_score(yva, yhatp))
    except Exception:
        auc = float(pd.Series(yva.values == yhat).mean())
    from sklearn.metrics import accuracy_score
    acc = float(accuracy_score(yva, yhat))

    artifact = pickle.dumps({"clf": clf, "features": ["return","rsi","volatility"]})
    meta = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "metric": float(auc),
        "metric_name": "roc_auc",
        "accuracy": float(acc),
        "n_samples": int(len(X)),
    }
    return artifact, meta

def _fit_trading_bot(sym_list: List[str], tfs: List[int], lookback_min: int) -> tuple[bytes, dict]:
    n_est = _per_strat_int("N_ESTIMATORS", "trading_bot", N_ESTIMATORS_DEFAULT)
    max_d = _per_strat_opt_int("MAX_DEPTH", "trading_bot", None if MAX_DEPTH_DEFAULT=="" else int(MAX_DEPTH_DEFAULT))
    return _fit_classifier(sym_list, tfs, lookback_min, n_est, max_d)

def _fit_ml_merged(sym_list: List[str], tfs: List[int], lookback_min: int) -> tuple[bytes, dict]:
    n_est = _per_strat_int("N_ESTIMATORS", "ml_merged", N_ESTIMATORS_DEFAULT)
    max_d = _per_strat_opt_int("MAX_DEPTH", "ml_merged", None if MAX_DEPTH_DEFAULT=="" else int(MAX_DEPTH_DEFAULT))
    return _fit_classifier(sym_list, tfs, lookback_min, n_est, max_d)

def _fit_ranked_ml(sym_list: List[str], tfs: List[int], lookback_min: int) -> tuple[bytes, dict]:
    n_est = _per_strat_int("N_ESTIMATORS", "ranked_ml", N_ESTIMATORS_DEFAULT)
    max_d = _per_strat_opt_int("MAX_DEPTH", "ranked_ml", None if MAX_DEPTH_DEFAULT=="" else int(MAX_DEPTH_DEFAULT))
    return _fit_classifier(sym_list, tfs, lookback_min, n_est, max_d)

def _fit_ml_sentiment(sym_list: List[str], tfs: List[int], lookback_min: int) -> tuple[bytes, dict]:
    n_est = _per_strat_int("N_ESTIMATORS", "ml_sentiment", N_ESTIMATORS_DEFAULT)
    max_d = _per_strat_opt_int("MAX_DEPTH", "ml_sentiment", None if MAX_DEPTH_DEFAULT=="" else int(MAX_DEPTH_DEFAULT))
    return _fit_classifier(sym_list, tfs, lookback_min, n_est, max_d)

# =========================================================
# REGISTRY + PROMOTION
# =========================================================

_registry = S3ModelRegistry(S3_BUCKET, S3_REGION, S3_BASE_PREFIX)

def _save_and_maybe_promote(strategy: str, artifact: bytes, meta: dict):
    cand_id = _registry.save_candidate(strategy, artifact, meta)

    # Per-strategy quality floor (if set)
    floor = _per_strat_float("PROMOTE_MIN", strategy, PROMOTE_MIN_DEFAULT)
    metric_value = float(meta.get("metric", 0.0))
    if metric_value < floor:
        print(f"[TRAIN] {strategy} metric={metric_value:.6f} < floor {floor:.6f} → NO PROMOTE  cand_id={cand_id}", flush=True)
        return False, cand_id

    promoted = _registry.compare_and_maybe_promote(strategy, cand_id, higher_is_better=True)
    print(f"[TRAIN] {strategy} metric={metric_value:.6f} samples={meta.get('n_samples',0)} promoted={promoted} cand_id={cand_id}", flush=True)
    return promoted, cand_id

def train_once(strategy: str, sym_list: List[str]):
    try:
        if strategy == "trading_bot":
            art, meta = _fit_trading_bot(sym_list, TRAIN_TFS, LOOKBACK_MIN)
        elif strategy == "ml_merged":
            art, meta = _fit_ml_merged(sym_list, TRAIN_TFS, LOOKBACK_MIN)
        elif strategy == "ranked_ml":
            art, meta = _fit_ranked_ml(sym_list, TRAIN_TFS, LOOKBACK_MIN)
        elif strategy == "ml_sentiment":
            art, meta = _fit_ml_sentiment(sym_list, TRAIN_TFS, LOOKBACK_MIN)
        else:
            # Unknown strategy: fall back to generic classifier
            art, meta = _fit_classifier(sym_list, TRAIN_TFS, LOOKBACK_MIN,
                                        N_ESTIMATORS_DEFAULT,
                                        None if MAX_DEPTH_DEFAULT=="" else int(MAX_DEPTH_DEFAULT))

        _save_and_maybe_promote(strategy, art, meta)

    except Exception as e:
        print(f"[TRAIN-ERROR] {strategy}: {e}", flush=True)

# =========================================================
# MAIN LOOP
# =========================================================

def main():
    if not (ALPACA_KEY and ALPACA_SECRET):
        print("[BOOT] Missing Alpaca creds; set ALPACA_KEY_ID / ALPACA_SECRET_KEY", flush=True)
    if not (S3_BUCKET and S3_REGION):
        print("[BOOT] Missing S3 config; set S3_BUCKET / S3_REGION", flush=True)

    print(f"[TRAINER] start strategies={TRAIN_STRATEGIES} tfs={TRAIN_TFS} use_universe={int(TRAIN_USE_UNIVERSE)}", flush=True)

    ran_today = False
    while True:
        try:
            now_hm = _now_et_hhmm()
            within = _within_window()
            force  = TRAIN_FORCE_RUN

            print(f"[TRAINER] tick now={now_hm} within_window={int(within)} force={int(force)} ran_today={int(ran_today)}", flush=True)

            if (within and not ran_today) or force:
                if force:
                    print("[TRAINER] TRAIN_FORCE_RUN=1 → running once now", flush=True)

                # (1) Build tonight's universe (assets-only)
                rebuild_universe_to_s3(_registry)

                # (2) Load symbols for training
                if TRAIN_USE_UNIVERSE:
                    sym_list = load_universe_from_s3(_registry)
                    if TRAIN_MAX_SYMBOLS > 0:
                        sym_list = sym_list[:TRAIN_MAX_SYMBOLS]
                else:
                    # fallback (rare)
                    sym_list = [s.strip().upper() for s in os.getenv("TRAIN_SYMBOLS","SPY,QQQ,AAPL,MSFT,NVDA").split(",") if s.strip()]

                print(f"[TRAIN] using {len(sym_list)} symbols", flush=True)

                # (3) Train & (maybe) promote per strategy
                for strat in TRAIN_STRATEGIES:
                    train_once(strat, sym_list)

                ran_today = True

            if not within:
                ran_today = False

        except Exception as e:
            print(f"[LOOP ERROR] {e}", flush=True)

        time.sleep(TRAINER_SLEEP_SEC)

if __name__ == "__main__":
    main()
