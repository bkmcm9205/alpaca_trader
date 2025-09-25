# app/trainer_worker.py
from __future__ import annotations

import os
import time
import json
import pickle
import math
from typing import List, Dict

import requests
import pandas as pd
from datetime import datetime
from zoneinfo import ZoneInfo
import boto3
from botocore.exceptions import ClientError

from app.data_alpaca import fetch_alpaca_1m
from app.model_registry_s3 import S3ModelRegistry

# ============================================
#              ENV & CONSTANTS
# ============================================

# --- Alpaca endpoints & creds (trainer needs data access) ---
ALPACA_TRADING_BASE = os.getenv("ALPACA_TRADING_BASE", "https://paper-api.alpaca.markets").rstrip("/")
ALPACA_DATA_BASE    = os.getenv("ALPACA_DATA_BASE", "https://data.alpaca.markets").rstrip("/")
ALPACA_KEY          = os.getenv("ALPACA_KEY_ID", os.getenv("APCA_API_KEY_ID", ""))
ALPACA_SECRET       = os.getenv("ALPACA_SECRET_KEY", os.getenv("APCA_API_SECRET_KEY", ""))

# --- S3 registry (trainer needs read/write) ---
S3_BUCKET      = os.getenv("S3_BUCKET", "")
S3_REGION      = os.getenv("S3_REGION", "")
S3_BASE_PREFIX = os.getenv("S3_BASE_PREFIX", "models")

# --- Strategies to train (names must match your adapters/registry) ---
TRAIN_STRATEGIES = [s.strip() for s in os.getenv("TRAIN_STRATEGIES", "ranked_ml,ml_merged").split(",") if s.strip()]

# --- Training universe & TFs (used by the toy trainer/backtest) ---
TRAIN_SYMBOLS = [s.strip().upper() for s in os.getenv("TRAIN_SYMBOLS", "SPY,QQQ,AAPL,MSFT,NVDA").split(",") if s.strip()]
TRAIN_TFS     = [int(x) for x in os.getenv("TRAIN_TFS", "5,15").split(",") if x.strip()]

# --- After-hours training window (ET) ---
RUN_START_ET = os.getenv("TRAIN_RUN_START_ET", "19:00")  # inclusive
RUN_END_ET   = os.getenv("TRAIN_RUN_END_ET",   "20:00")  # inclusive
TRAINER_SLEEP_SEC = int(os.getenv("TRAINER_SLEEP_SEC", "60"))

# --- Metric we optimize (higher is better) ---
METRIC_NAME = os.getenv("TRAIN_METRIC", "score")

# --- Nightly universe builder (writes the list your traders load pre-market) ---
UNIVERSE_WRITE_KEY = os.getenv("UNIVERSE_S3_WRITE_KEY", "models/universe/daily.csv")
UNIVERSE_BATCH     = int(os.getenv("UNIVERSE_BATCH", "400"))   # symbols per snapshots request
UNIVERSE_MAX       = int(os.getenv("UNIVERSE_MAX", "8000"))    # cap to be safe

# Universe filter knobs (defaults align with your prior scanner-style thresholds)
U_MIN_PRICE     = float(os.getenv("UNIVERSE_MIN_LAST_PRICE", os.getenv("MIN_LAST_PRICE", "3")))
U_MIN_AVG_VOL   = int(os.getenv("UNIVERSE_MIN_AVG_VOL",      os.getenv("SCANNER_MIN_AVG_VOL", "200000")))
U_MIN_TODAY_VOL = int(os.getenv("UNIVERSE_MIN_TODAY_VOL",    os.getenv("SCANNER_MIN_TODAY_VOL", "100000")))

# ============================================
#                UTILITIES
# ============================================

def _now_et_hhmm() -> str:
    return datetime.now(ZoneInfo("America/New_York")).strftime("%H:%M")

def _within_window() -> bool:
    """Return True if current ET time is within RUN_START_ET..RUN_END_ET (inclusive)."""
    now_hm = _now_et_hhmm()
    return RUN_START_ET <= now_hm <= RUN_END_ET

def _auth_headers() -> Dict[str, str]:
    return {
        "Apca-Api-Key-Id": ALPACA_KEY,
        "Apca-Api-Secret-Key": ALPACA_SECRET,
        "accept": "application/json",
    }

# ============================================
#          NIGHTLY UNIVERSE BUILDER
# ============================================

def _list_us_equities() -> List[str]:
    """List active, tradable US equities from Alpaca /v2/assets."""
    url = f"{ALPACA_TRADING_BASE}/v2/assets"
    params = {"asset_class": "us_equity", "status": "active"}
    r = requests.get(url, headers=_auth_headers(), params=params, timeout=60)
    r.raise_for_status()
    assets = r.json()
    syms = [a["symbol"].upper() for a in assets if a.get("tradable")]
    return syms[:UNIVERSE_MAX]

def _snapshots_batch(symbols: List[str]) -> dict:
    """Fetch multi-symbol snapshots from Alpaca data API (SIP feed)."""
    if not symbols:
        return {}
    # GET /v2/stocks/snapshots?symbols=AAPL,MSFT,...&feed=sip
    url = f"{ALPACA_DATA_BASE}/v2/stocks/snapshots"
    params = {"symbols": ",".join(symbols), "feed": "sip"}
    r = requests.get(url, headers=_auth_headers(), params=params, timeout=60)
    r.raise_for_status()
    return r.json().get("snapshots", {})

def rebuild_universe_to_s3(registry: S3ModelRegistry):
    """Build a nightly symbol list by filtering snapshots; write to S3 as newline CSV."""
    if not (S3_BUCKET and S3_REGION and UNIVERSE_WRITE_KEY):
        print("[UNIVERSE] S3 not configured; skipping", flush=True)
        return

    try:
        symbols = _list_us_equities()
    except Exception as e:
        print(f"[UNIVERSE] error listing assets: {e}", flush=True)
        return

    if not symbols:
        print("[UNIVERSE] No assets returned", flush=True)
        return

    keep: List[str] = []
    for i in range(0, len(symbols), UNIVERSE_BATCH):
        batch = symbols[i:i+UNIVERSE_BATCH]
        try:
            snaps = _snapshots_batch(batch)
        except Exception as e:
            print(f"[UNIVERSE] snapshots error on batch {i//UNIVERSE_BATCH+1}: {e}", flush=True)
            time.sleep(0.5)
            continue

        for sym in batch:
            s = snaps.get(sym)
            if not s:
                continue
            # Prefer previous daily bar to avoid partial intraday data
            prev = (s.get("prevDailyBar") or {})  # {o,h,l,c,v,t}
            last_close = float(prev.get("c") or 0.0)
            prev_vol   = int(prev.get("v") or 0)
            # Optional: require some minimum current-day volume too
            daily = (s.get("dailyBar") or {})
            today_vol = int(daily.get("v") or 0)

            if last_close >= U_MIN_PRICE and prev_vol >= U_MIN_AVG_VOL and today_vol >= U_MIN_TODAY_VOL:
                keep.append(sym)

        print(f"[UNIVERSE] batch {i//UNIVERSE_BATCH+1}: kept_so_far={len(keep)}", flush=True)
        time.sleep(0.25)  # be nice to the API even though Algo Plus allows high rate

    if not keep:
        print("[UNIVERSE] Filter produced 0 symbols; writing fallback SPY,QQQ", flush=True)
        keep = ["SPY", "QQQ"]

    # Write newline-delimited list to S3
    try:
        csv_bytes = ("\n".join(sorted(set(keep))) + "\n").encode("utf-8")
        registry.s3.put_object(Bucket=S3_BUCKET, Key=UNIVERSE_WRITE_KEY, Body=csv_bytes)
        print(f"[UNIVERSE] wrote {len(set(keep))} symbols to s3://{S3_BUCKET}/{UNIVERSE_WRITE_KEY}", flush=True)
    except ClientError as e:
        print(f"[UNIVERSE] S3 write error: {e}", flush=True)

# ============================================
#          SIMPLE TRAIN / PROMOTE LOOP
# ============================================

_registry = S3ModelRegistry(S3_BUCKET, S3_REGION, S3_BASE_PREFIX)

def _aggregate_bars(symbol: str, tf_min: int, lookback_min: int = 7*24*60) -> pd.DataFrame:
    """Fetch 1m bars then resample to tf_min; simple OHLCV aggregation."""
    df = fetch_alpaca_1m(symbol, limit=lookback_min)
    if df is None or df.empty:
        return pd.DataFrame()
    if tf_min <= 1:
        return df
    rule = f"{tf_min}T"
    agg = df.resample(rule).agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum"
    }).dropna()
    return agg

def _toy_fit(symbols: List[str], tfs: List[int]) -> tuple[bytes, dict]:
    """
    Placeholder trainer:
      - computes a naive momentum-ish metric across symbols/TFs
      - returns (artifact_bytes, meta_dict)
    Replace with your real training code that serializes your model.
    """
    metrics: List[float] = []
    for sym in symbols:
        for tf in tfs:
            d = _aggregate_bars(sym, tf)
            if d is None or d.empty:
                continue
            r = d["close"].pct_change().dropna()
            # simple win/loss imbalance proxy
            score = float((r[r > 0].mean() - r[r < 0].abs().mean()) or 0.0)
            if not math.isnan(score):
                metrics.append(score)

    metric = float(pd.Series(metrics).mean()) if metrics else 0.0
    # Example artifact: JSON; swap with your pickled model, Torch, ONNX, etc.
    model_json = {"kind": "toy_momentum", "trained_on": {"symbols": symbols, "tfs": tfs}, "params": {}}
    artifact = json.dumps(model_json).encode("utf-8")
    meta = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "metric": metric,
        "metric_name": METRIC_NAME,
    }
    return artifact, meta

def train_once(strategy: str):
    """Train one strategy; save candidate; promote if better than production."""
    try:
        art, meta = _toy_fit(TRAIN_SYMBOLS, TRAIN_TFS)
        cand_id = _registry.save_candidate(strategy, art, meta)
        promoted = _registry.compare_and_maybe_promote(strategy, cand_id, higher_is_better=True)
        print(f"[TRAIN] {strategy} metric={meta['metric']:.6f} promoted={promoted} cand_id={cand_id}", flush=True)
    except Exception as e:
        print(f"[TRAIN-ERROR] {strategy}: {e}", flush=True)

def main():
    # Basic sanity checks
    if not (ALPACA_KEY and ALPACA_SECRET):
        print("[BOOT] Missing Alpaca creds; set ALPACA_KEY_ID / ALPACA_SECRET_KEY", flush=True)
    if not (S3_BUCKET and S3_REGION):
        print("[BOOT] Missing S3 config; set S3_BUCKET / S3_REGION", flush=True)

    print(f"[TRAINER] starting strategies={TRAIN_STRATEGIES} symbols={TRAIN_SYMBOLS} tfs={TRAIN_TFS}", flush=True)

    ran_today = False
    while True:
        try:
            if _within_window() and not ran_today:
                # 1) Build tonight's universe file for traders to load pre-market
                rebuild_universe_to_s3(_registry)

                # 2) Train & (maybe) promote for each ML strategy
                for strat in TRAIN_STRATEGIES:
                    train_once(strat)

                ran_today = True

            # Reset the daily latch after we exit the window
            if not _within_window():
                ran_today = False

        except Exception as e:
            print(f"[LOOP ERROR] {e}", flush=True)

        time.sleep(TRAINER_SLEEP_SEC)

if __name__ == "__main__":
    main()
