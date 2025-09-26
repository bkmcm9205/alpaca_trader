# app/strategy_runner.py
from __future__ import annotations
import os, time, importlib
from dataclasses import dataclass
from typing import Dict, List
from collections import defaultdict

import pandas as pd
import boto3
from botocore.exceptions import ClientError
from datetime import datetime
from zoneinfo import ZoneInfo

from app.data_alpaca import fetch_alpaca_1m
from app.broker_alpaca import (
    place_bracket_order,
    get_positions_symbols,
    get_account_summary,
)
from app.strategy_registry import ADAPTERS

# =============================
#         ENV / CONFIG
# =============================

S3_BUCKET = os.getenv("S3_BUCKET","")
S3_REGION = os.getenv("S3_REGION","")
UNIVERSE_S3_KEY = os.getenv("UNIVERSE_S3_KEY","")
UNIVERSE_RELOAD_WINDOW_ET = os.getenv("UNIVERSE_RELOAD_WINDOW_ET","08:00-09:25")
UNIVERSE_REFRESH_SEC = int(os.getenv("UNIVERSE_REFRESH_SEC","60"))

SYMBOLS = [s.strip().upper() for s in os.getenv("SCANNER_SYMBOLS","").split(",") if s.strip()]
TF_MIN_LIST = [int(x) for x in os.getenv("TF_MIN_LIST","5,15").split(",") if x.strip()]

ACTIVE = [s.strip() for s in os.getenv("ACTIVE_STRATEGIES","trading_bot,ml_merged,ranked_ml").split(",") if s.strip()]

LOOKBACK_MIN = int(os.getenv("LOOKBACK_MIN","2400"))
POLL_SECONDS = int(os.getenv("POLL_SECONDS","20"))

DRY_RUN = os.getenv("DRY_RUN","1") in ("1","true","True")
DEDUP_BY_SYMBOL = os.getenv("DEDUP_BY_SYMBOL","1") in ("1","true","True")
SKIP_IF_POSITION_OPEN = os.getenv("SKIP_IF_POSITION_OPEN","1") in ("1","true","True")

RANKED_STRATS = [s.strip() for s in os.getenv("RANKED_STRATS","ranked_ml").split(",") if s.strip()]
TOPK_RANKED_DEFAULT = int(os.getenv("TOPK_RANKED_DEFAULT","1"))
CONF_THR_DEFAULT = float(os.getenv("CONF_THR_DEFAULT","0.05"))
MIN_QTY_DEFAULT = int(os.getenv("MIN_QTY_DEFAULT","1"))
_NONRANK_BIG = 10**9

# --- NEW: Buying power & RTH guards ---
BP_RESERVE_PCT = float(os.getenv("BP_RESERVE_PCT", "0.10"))
MIN_NOTIONAL_PER_ORDER = float(os.getenv("MIN_NOTIONAL_PER_ORDER", "50"))
REQUIRE_RTH = os.getenv("REQUIRE_RTH", "0") in ("1","true","True")

def _per_strat(key: str, strat: str, cast, default):
    env_key = f"{key}_{strat.upper()}"
    v = os.getenv(env_key)
    return cast(v) if v is not None else default

# =============================
#   PRE-MARKET UNIVERSE LOADER
# =============================

_universe_syms: List[str] = []
_universe_loaded_day = None
_universe_last_check = 0.0

def _in_window(window: str) -> bool:
    try:
        s,e = window.split("-",1)
        now_hm = datetime.now(ZoneInfo("America/New_York")).strftime("%H:%M")
        return s <= now_hm <= e
    except Exception:
        return False

def _load_universe(force_boot: bool = False):
    global _universe_syms, _universe_loaded_day, _universe_last_check
    if not (S3_BUCKET and S3_REGION and UNIVERSE_S3_KEY):
        return

    def _clean_sym(s: str) -> str:
        return s.replace("\ufeff","").strip().upper()

    now = time.time()
    if force_boot and not _universe_syms:
        pass
    else:
        if (now - _universe_last_check) < UNIVERSE_REFRESH_SEC:
            return
        _universe_last_check = now
        if not _in_window(UNIVERSE_RELOAD_WINDOW_ET):
            return
        today = datetime.now(ZoneInfo("America/New_York")).date()
        if _universe_loaded_day == today:
            return

    try:
        s3 = boto3.client(
            "s3",
            region_name=S3_REGION,
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        )
        body = s3.get_object(Bucket=S3_BUCKET, Key=UNIVERSE_S3_KEY)["Body"].read().decode("utf-8","ignore")
        syms = [ln.replace("\ufeff","").strip().upper()
                for ln in body.splitlines() if ln.strip() and not ln.startswith("#")]
        if syms:
            _universe_syms = syms
            _universe_loaded_day = datetime.now(ZoneInfo("America/New_York")).date()
            print(f"[UNIVERSE] loaded {len(syms)} symbols from s3://{S3_BUCKET}/{UNIVERSE_S3_KEY}", flush=True)
    except ClientError as e:
        print(f"[UNIVERSE] load error: {e}", flush=True)

# =============================
#         STRAT ADAPTERS
# =============================

@dataclass
class Candidate:
    strategy: str
    symbol: str
    tf: int
    confidence: float
    side: str
    qty: int
    tp: float|None
    sl: float|None
    last: float

def _load_adapters():
    mods = {}
    for name in ACTIVE:
        path = ADAPTERS.get(name)
        if path:
            mods[name] = importlib.import_module(path)
        else:
            print(f"[BOOT] WARNING: no adapter for '{name}'", flush=True)
    return mods

def _norm_qty(q, strat):
    try:
        q = int(q or 0)
    except Exception:
        q = 0
    return q if q>0 else _per_strat("MIN_QTY", strat, int, MIN_QTY_DEFAULT)

# =============================
#         SCAN & RANK
# =============================

def scan_once(adapters) -> List[Candidate]:
    cands: List[Candidate] = []
    open_syms = get_positions_symbols() if SKIP_IF_POSITION_OPEN else set()

    _load_universe(force_boot=True)
    symbols = _universe_syms or SYMBOLS or ["SPY","QQQ"]

    for sym in symbols:
        df1m = fetch_alpaca_1m(sym, limit=LOOKBACK_MIN)
        if df1m is None or df1m.empty: 
            continue
        last = float(df1m["close"].iloc[-1])

        for tf in TF_MIN_LIST:
            for strat, mod in adapters.items():
                dec = getattr(mod,"decide",None)
                if not dec: continue
                try:
                    sig = dec(sym, df1m, tf)
                except Exception as e:
                    print(f"[{strat}] decide error {sym}:{tf}m -> {e}", flush=True)
                    continue
                if not sig: continue
                conf = float(sig.get("confidence",0.0))
                thr = _per_strat("CONF_THR", strat, float, CONF_THR_DEFAULT)
                if conf < thr: continue
                side = str(sig.get("side","")).lower()
                if side not in ("buy","sell"): continue
                if SKIP_IF_POSITION_OPEN and side=="buy" and sym in open_syms:
                    continue
                qty = _norm_qty(sig.get("quantity"), strat)
                tp  = sig.get("takeProfit")
                sl  = sig.get("stopLoss")
                cands.append(Candidate(strat,sym,tf,conf,side,qty,tp,sl,last))
    return cands

def _is_rth_now() -> bool:
    now = datetime.now(ZoneInfo("America/New_York"))
    h,m = now.hour, now.minute
    return ((h>9) or (h==9 and m>=30)) and (h<16)

# =============================
#     RANK, GUARD, EXECUTE
# =============================

def rank_and_execute(candidates: List[Candidate]):
    if not candidates:
        print("[RANK] no candidates", flush=True)
        return

    grouped: Dict[str,List[Candidate]] = defaultdict(list)
    for c in candidates:
        grouped[c.strategy].append(c)

    orders: List[Candidate] = []

    for strat, items in grouped.items():
        ranked_mode = strat in RANKED_STRATS
        if DEDUP_BY_SYMBOL:
            best_by_symbol: Dict[str,Candidate] = {}
            for c in items:
                cur = best_by_symbol.get(c.symbol)
                if cur is None or c.confidence > cur.confidence:
                    best_by_symbol[c.symbol] = c
            items = list(best_by_symbol.values())
        items.sort(key=lambda x: x.confidence, reverse=True)
        if ranked_mode:
            topk = _per_strat("TOPK", strat, int, TOPK_RANKED_DEFAULT)
            take = items[:max(0,topk)]
            print(f"[RANK] {strat}: {len(items)} → taking {len(take)}", flush=True)
            orders.extend(take)
        else:
            topk = _per_strat("TOPK", strat, int, _NONRANK_BIG)
            take = items[:max(0,topk)]
            print(f"[PASS] {strat}: {len(items)} → passing {len(take)}", flush=True)
            orders.extend(take)

    if REQUIRE_RTH and not _is_rth_now():
        print("[SKIP] Outside RTH; suppressing entries.", flush=True)
        return

    try:
        acct = get_account_summary()
        bp_total = float(acct.get("buying_power") or 0.0)
    except Exception:
        bp_total = 0.0

    bp_avail = max(0.0, bp_total * (1.0 - BP_RESERVE_PCT))

    for o in orders:
        notional = float(o.qty) * float(o.last)
        if notional < max(1.0, MIN_NOTIONAL_PER_ORDER):
            pass
        elif notional > bp_avail:
            print(f"[SKIP] Low BP: need ~{notional:.2f}, have ~{bp_avail:.2f} → {o.symbol} skipped", flush=True)
            continue

        if DRY_RUN:
            print(f"[DRY] {o.strategy} conf={o.confidence:.4f} {o.symbol}:{o.tf}m side={o.side} qty={o.qty} tp={o.tp} sl={o.sl}", flush=True)
        else:
            ok, info = place_bracket_order(o.symbol, o.side, o.qty, o.last, o.tp, o.sl)
            print(f"[ORDER] {o.strategy} {o.symbol}:{o.tf}m conf={o.confidence:.4f} side={o.side} qty={o.qty} ok={ok} info={str(info)[:200]}", flush=True)
            if ok:
                bp_avail = max(0.0, bp_avail - notional)

# =============================
#             MAIN
# =============================

def main():
    adapters = _load_adapters()
    print(f"[BOOT] ACTIVE={list(adapters.keys())} TFs={TF_MIN_LIST} DRY_RUN={int(DRY_RUN)} RANKED_STRATS={RANKED_STRATS}", flush=True)
    if UNIVERSE_S3_KEY:
        print(f"[BOOT] Using S3 universe: s3://{S3_BUCKET}/{UNIVERSE_S3_KEY} (reload window {UNIVERSE_RELOAD_WINDOW_ET})", flush=True)
    elif SYMBOLS:
        print(f"[BOOT] Using SCANNER_SYMBOLS={SYMBOLS}", flush=True)
    else:
        print("[BOOT] No universe set; fallback to ['SPY','QQQ']", flush=True)

    while True:
        start = time.time()
        try:
            cands = scan_once(adapters)
            rank_and_execute(cands)
        except Exception as e:
            print(f"[LOOP ERROR] {e}", flush=True)
        elapsed = time.time() - start
        time.sleep(max(1, POLL_SECONDS - int(elapsed)))

if __name__ == "__main__":
    main()
