# app/trainer_worker.py
from __future__ import annotations

import os
import time
import json
import math
from typing import List, Dict
from datetime import datetime
from zoneinfo import ZoneInfo

import requests
import pandas as pd
import boto3
from botocore.exceptions import ClientError

from app.data_alpaca import fetch_alpaca_1m
from app.model_registry_s3 import S3ModelRegistry

# =========================
# ENV & CONSTANTS
# =========================

ALPACA_TRADING_BASE = os.getenv("ALPACA_TRADING_BASE", "https://paper-api.alpaca.markets").rstrip("/")
ALPACA_DATA_BASE    = os.getenv("ALPACA_DATA_BASE", "https://data.alpaca.markets").rstrip("/")
ALPACA_KEY          = os.getenv("ALPACA_KEY_ID", os.getenv("APCA_API_KEY_ID", ""))
ALPACA_SECRET       = os.getenv("ALPACA_SECRET_KEY", os.getenv("APCA_API_SECRET_KEY", ""))

S3_BUCKET      = os.getenv("S3_BUCKET", "")
S3_REGION      = os.getenv("S3_REGION", "")
S3_BASE_PREFIX = os.getenv("S3_BASE_PREFIX", "models")

TRAIN_STRATEGIES = [s.strip() for s in os.getenv("TRAIN_STRATEGIES", "ranked_ml,ml_merged").split(",") if s.strip()]
TRAIN_TFS        = [int(x) for x in os.getenv("TRAIN_TFS", "5,15").split(",") if x.strip()]
TRAIN_METRIC     = os.getenv("TRAIN_METRIC", "score")

RUN_START_ET = os.getenv("TRAIN_RUN_START_ET", "19:00")
RUN_END_ET   = os.getenv("TRAIN_RUN_END_ET",   "20:00")
TRAINER_SLEEP_SEC = int(os.getenv("TRAINER_SLEEP_SEC", "60"))

# Use the S3 universe for training
TRAIN_USE_UNIVERSE = os.getenv("TRAIN_USE_UNIVERSE", "1").lower() in ("1","true","yes")
TRAIN_MAX_SYMBOLS  = int(os.getenv("TRAIN_MAX_SYMBOLS", "0"))  # 0 = no cap

# Universe output
UNIVERSE_WRITE_KEY = os.getenv("UNIVERSE_S3_WRITE_KEY", "models/universe/daily.csv")
UNIVERSE_MAX       = int(os.getenv("UNIVERSE_MAX", "8000"))

# =========================
# UTIL
# =========================

def _now_et_hhmm() -> str:
    return datetime.now(ZoneInfo("America/New_York")).strftime("%H:%M")

def _within_window() -> bool:
    now_hm = _now_et_hhmm()
    return RUN_START_ET <= now_hm <= RUN_END_ET

def _auth_headers() -> Dict[str, str]:
    return {"Apca-Api-Key-Id": ALPACA_KEY, "Apca-Api-Secret-Key": ALPACA_SECRET, "accept": "application/json"}

# =========================
# UNIVERSE (assets-only)
# =========================

def _list_us_equities() -> List[str]:
    url = f"{ALPACA_TRADING_BASE}/v2/assets"
    params = {"asset_class": "us_equity", "status": "active"}
    r = requests.get(url, headers=_auth_headers(), params=params, timeout=60)
    r.raise_for_status()
    assets = r.json()
    syms = [str(a.get("symbol","")).upper().strip() for a in assets if a.get("tradable")]
    return syms[:UNIVERSE_MAX]

def rebuild_universe_to_s3(registry: S3ModelRegistry):
    """Assets-only mode: write all active, tradable US equities to S3."""
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

# =========================
# SIMPLE TRAIN / PROMOTE
# =========================

_registry = S3ModelRegistry(S3_BUCKET, S3_REGION, S3_BASE_PREFIX)

def _aggregate_bars(sym: str, tf_min: int, lookback_min: int = 7*24*60) -> pd.DataFrame:
    df = fetch_alpaca_1m(sym, limit=lookback_min)
    if df is None or df.empty:
        return pd.DataFrame()
    if tf_min <= 1:
        return df
    rule = f"{tf_min}min"   # 'T' deprecated; use 'min'
    agg = df.resample(rule).agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum"
    }).dropna()
    return agg

def _toy_fit(sym_list: List[str], tfs: List[int]) -> tuple[bytes, dict]:
    metrics: List[float] = []
    for sym in sym_list:
        for tf in tfs:
            d = _aggregate_bars(sym, tf)
            if d is None or d.empty:
                continue
            r = d["close"].pct_change().dropna()
            score = float((r[r > 0].mean() - r[r < 0].abs().mean()) or 0.0)
            if not math.isnan(score):
                metrics.append(score)

    metric = float(pd.Series(metrics).mean()) if metrics else 0.0
    model_json = {"kind": "toy_momentum", "trained_on": {"symbols_count": len(sym_list), "tfs": tfs}, "params": {}}
    artifact = json.dumps(model_json).encode("utf-8")
    meta = {"ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "metric": metric, "metric_name": TRAIN_METRIC}
    return artifact, meta

def train_once(strategy: str, sym_list: List[str]):
    try:
        art, meta = _toy_fit(sym_list, TRAIN_TFS)
        cand_id = _registry.save_candidate(strategy, art, meta)
        promoted = _registry.compare_and_maybe_promote(strategy, cand_id, higher_is_better=True)
        print(f"[TRAIN] {strategy} metric={meta['metric']:.6f} on {len(sym_list)} syms promoted={promoted} cand_id={cand_id}", flush=True)
    except Exception as e:
        print(f"[TRAIN-ERROR] {strategy}: {e}", flush=True)

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
            force  = os.getenv("TRAIN_FORCE_RUN", "0").lower() in ("1","true","yes")

            print(f"[TRAINER] tick now={now_hm} within_window={int(within)} force={int(force)} ran_today={int(ran_today)}", flush=True)

            if (within and not ran_today) or force:
                if force:
                    print("[TRAINER] FORCE_RUN=1 -> running once now", flush=True)

                # (1) Build tonight's universe (assets-only)
                rebuild_universe_to_s3(_registry)

                # (2) Load symbols for training
                if TRAIN_USE_UNIVERSE:
                    sym_list = load_universe_from_s3(_registry)
                    if TRAIN_MAX_SYMBOLS > 0:
                        sym_list = sym_list[:TRAIN_MAX_SYMBOLS]
                else:
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
