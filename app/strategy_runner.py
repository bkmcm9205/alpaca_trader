# app/strategy_runner.py
from __future__ import annotations

import os, time, importlib, math
from dataclasses import dataclass
from typing import Dict, List
from collections import defaultdict

import pandas as pd
import boto3
from botocore.exceptions import ClientError, BotoCoreError
from datetime import datetime
from zoneinfo import ZoneInfo

from app.data_alpaca import fetch_alpaca_1m
from app.broker_alpaca import place_bracket_order, get_positions_symbols
from app.strategy_registry import ADAPTERS

# Optional: force model preload at boot so you see "[MODEL:...] loaded ..." once
try:
    from app.strategies.model_runtime import maybe_load_model
except Exception:
    maybe_load_model = None

# =============================
#         ENV / CONFIG
# =============================

# S3-driven symbol universe (loaded pre-market only; otherwise static all day)
S3_BUCKET = os.getenv("S3_BUCKET","")
S3_REGION = os.getenv("S3_REGION","")
UNIVERSE_S3_KEY = os.getenv("UNIVERSE_S3_KEY","")  # e.g., models/universe/daily.csv
UNIVERSE_RELOAD_WINDOW_ET = os.getenv("UNIVERSE_RELOAD_WINDOW_ET","08:00-09:25")
UNIVERSE_REFRESH_SEC = int(os.getenv("UNIVERSE_REFRESH_SEC","60"))

# Scan/timeframe config
SYMBOLS = [s.strip().upper() for s in os.getenv("SCANNER_SYMBOLS","").split(",") if s.strip()]
TF_MIN_LIST = [int(x) for x in os.getenv("TF_MIN_LIST","5,15").split(",") if x.strip()]

ACTIVE = [s.strip() for s in os.getenv("ACTIVE_STRATEGIES","trading_bot,ml_merged,ranked_ml").split(",") if s.strip()]

LOOKBACK_MIN = int(os.getenv("LOOKBACK_MIN","2400"))     # ~40h of 1m bars
POLL_SECONDS = int(os.getenv("POLL_SECONDS","20"))

DRY_RUN = os.getenv("DRY_RUN","1") in ("1","true","True")
DEDUP_BY_SYMBOL = os.getenv("DEDUP_BY_SYMBOL","1") in ("1","true","True")
SKIP_IF_POSITION_OPEN = os.getenv("SKIP_IF_POSITION_OPEN","1") in ("1","true","True")

# Ranking applies only to these strategies (your core: ranked_ml)
RANKED_STRATS = [s.strip() for s in os.getenv("RANKED_STRATS","ranked_ml").split(",") if s.strip()]
TOPK_RANKED_DEFAULT = int(os.getenv("TOPK_RANKED_DEFAULT","1"))       # take top-K for ranked strats
CONF_THR_DEFAULT = float(os.getenv("CONF_THR_DEFAULT","0.05"))
MIN_QTY_DEFAULT = int(os.getenv("MIN_QTY_DEFAULT","1"))  # used only if risk-based can't compute qty
_NONRANK_BIG = 10**9  # effectively "no cap" for non-ranked unless TOPK_<STRAT> set

# ----- Risk-based sizing env (RESTORED like your old models) -----
EQUITY_USD  = float(os.getenv("EQUITY_USD",  "100000"))
RISK_PCT    = float(os.getenv("RISK_PCT",    "0.01"))   # 1% risk per trade
MAX_POS_PCT = float(os.getenv("MAX_POS_PCT", "0.05"))   # 5% notional cap per trade
MIN_QTY_ENV = int(os.getenv("MIN_QTY", "1"))
ROUND_LOT   = int(os.getenv("ROUND_LOT","1"))

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
    """Load newline-delimited symbols from S3 once per day (pre-market window only)."""
    global _universe_syms, _universe_loaded_day, _universe_last_check
    if not (S3_BUCKET and S3_REGION and UNIVERSE_S3_KEY):
        return

    def _clean_sym(s: str) -> str:
        # strip UTF-8 BOM + whitespace + normalize to UPPER
        return s.replace("\ufeff", "").strip().upper()

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
        body = s3.get_object(Bucket=S3_BUCKET, Key=UNIVERSE_S3_KEY)["Body"].read().decode("utf-8", errors="ignore")
        syms_raw = body.splitlines()
        syms = []
        for ln in syms_raw:
            sym = _clean_sym(ln)
            if not sym or sym.startswith("#"):
                continue
            syms.append(sym)
        if syms:
            _universe_syms = syms
            _universe_loaded_day = datetime.now(ZoneInfo("America/New_York")).date()
            print(f"[UNIVERSE] loaded {len(syms)} symbols from s3://{S3_BUCKET}/{UNIVERSE_S3_KEY}", flush=True)
    except (ClientError, BotoCoreError) as e:
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
    tp: float | None
    sl: float | None
    last: float

def _load_adapters():
    mods = {}
    for name in ACTIVE:
        path = ADAPTERS.get(name)
        if not path:
            print(f"[BOOT] WARNING: no adapter mapped for strategy '{name}'", flush=True)
            continue
        try:
            mods[name] = importlib.import_module(path)
        except Exception as e:
            print(f"[BOOT] ERROR: failed to import adapter '{name}' -> {e}", flush=True)
    return mods

# =============================
#     RISK-BASED SIZING (RESTORED)
# =============================

def _stop_price_from(last: float, sl_val: float | None, side: str) -> float | None:
    """
    sl_val may be percentage (<1) or absolute (>=1).
    For longs (buy): stop below entry. For shorts (sell): stop above entry.
    """
    if sl_val is None:
        return None
    try:
        v = float(sl_val)
    except Exception:
        return None
    if v <= 0:
        return None
    if v < 1.0:
        # percent input
        return last * (1.0 - v) if side == "buy" else last * (1.0 + v)
    else:
        # absolute input
        return float(v)

def _qty_from_risk(last: float, stop: float | None, equity=EQUITY_USD,
                   risk_pct=RISK_PCT, max_pos_pct=MAX_POS_PCT,
                   min_qty=MIN_QTY_ENV, round_lot=ROUND_LOT) -> int:
    if last is None or stop is None:
        return 0
    risk_per_share = abs(float(last) - float(stop))
    if risk_per_share <= 0:
        return 0
    qty_risk     = (equity * risk_pct) / risk_per_share
    qty_notional = (equity * max_pos_pct) / max(1e-9, float(last))
    qty = math.floor(max(min(qty_risk, qty_notional), 0) / max(1, round_lot)) * max(1, round_lot)
    if qty <= 0:
        return 0
    return int(max(qty, min_qty))

def _compute_qty(side: str, qty_signal, last: float, sl_val: float | None) -> int:
    """
    If strategy provided a positive quantity, honor it.
    Otherwise compute from risk using stopLoss and last price.
    Fallback to MIN_QTY_DEFAULT only if we cannot compute a risk-based size.
    """
    try:
        q = int(qty_signal or 0)
    except Exception:
        q = 0
    if q > 0:
        return q
    stop = _stop_price_from(last, sl_val, side)
    q_risk = _qty_from_risk(last, stop)
    if q_risk > 0:
        return q_risk
    return MIN_QTY_DEFAULT

# =============================
#         SCAN & RANK
# =============================

def scan_once(adapters) -> List[Candidate]:
    candidates: List[Candidate] = []
    open_syms = get_positions_symbols() if SKIP_IF_POSITION_OPEN else set()

    # Universe precedence: S3 universe → SCANNER_SYMBOLS → fallback
    _load_universe(force_boot=True)
    symbols = _universe_syms or SYMBOLS or ["SPY","QQQ"]

    for sym in symbols:
        # Fetch bars; quietly skip symbols that error (e.g., Alpaca 500s)
        try:
            df1m = fetch_alpaca_1m(sym, limit=LOOKBACK_MIN)
        except Exception:
            continue

        if df1m is None or df1m.empty:
            continue

        # last price
        try:
            last = float(df1m["close"].iloc[-1])
        except Exception:
            continue

        for tf in TF_MIN_LIST:
            for strat, mod in adapters.items():
                dec = getattr(mod, "decide", None)
                if not dec:
                    continue

                try:
                    sig = dec(sym, df1m, tf)
                except Exception as e:
                    print(f"[{strat}] decide error {sym}:{tf}m -> {e}", flush=True)
                    continue

                if not sig:
                    continue

                conf = float(sig.get("confidence", 0.0) or 0.0)
                thr  = _per_strat("CONF_THR", strat, float, CONF_THR_DEFAULT)
                if conf < thr:
                    continue

                side = str(sig.get("side","")).lower()
                if side not in ("buy","sell"):
                    continue

                if SKIP_IF_POSITION_OPEN and side == "buy" and sym in open_syms:
                    # long-only de-dupe; adjust if you also run shorts netting
                    continue

                # Risk-based sizing if strategy didn't set quantity
                qty = _compute_qty(side, sig.get("quantity"), last, sig.get("stopLoss"))

                tp = sig.get("takeProfit")
                sl = sig.get("stopLoss")

                candidates.append(Candidate(
                    strategy=strat, symbol=sym, tf=tf, confidence=conf,
                    side=side, qty=qty, tp=tp, sl=sl, last=last
                ))
    return candidates

def rank_and_execute(candidates: List[Candidate]):
    if not candidates:
        print("[RANK] no candidates", flush=True)
        return

    grouped: Dict[str, List[Candidate]] = defaultdict(list)
    for c in candidates:
        grouped[c.strategy].append(c)

    orders: List[Candidate] = []

    for strat, items in grouped.items():
        ranked_mode = strat in RANKED_STRATS

        # Optional de-dup to 1 per symbol (keep highest confidence)
        if DEDUP_BY_SYMBOL:
            best_by_symbol: Dict[str, Candidate] = {}
            for c in items:
                cur = best_by_symbol.get(c.symbol)
                if cur is None or c.confidence > cur.confidence:
                    best_by_symbol[c.symbol] = c
            items = list(best_by_symbol.values())

        # Sort by confidence (desc)
        items.sort(key=lambda x: x.confidence, reverse=True)

        if ranked_mode:
            topk = _per_strat("TOPK", strat, int, TOPK_RANKED_DEFAULT)
            take = items[:max(0, topk)]
            print(f"[RANK] {strat}: {len(items)} candidates -> taking top {len(take)}", flush=True)
            orders.extend(take)
        else:
            topk = _per_strat("TOPK", strat, int, _NONRANK_BIG)
            take = items[:max(0, topk)]
            print(f"[PASS] {strat}: {len(items)} candidates -> passing {len(take)}", flush=True)
            orders.extend(take)

    for o in orders:
        if DRY_RUN:
            print(f"[DRY] {o.strategy} conf={o.confidence:.4f} {o.symbol}:{o.tf}m side={o.side} qty={o.qty} tp={o.tp} sl={o.sl}", flush=True)
        else:
            ok, info = place_bracket_order(o.symbol, o.side, o.qty, o.last, o.tp, o.sl)
            print(f"[ORDER] {o.strategy} {o.symbol}:{o.tf}m conf={o.confidence:.4f} side={o.side} qty={o.qty} ok={ok} info={str(info)[:160]}", flush=True)

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

    # Optional: preload models so loader logs once at boot
    if maybe_load_model:
        try:
            print(f"[BOOT] Preloading models for: {list(adapters.keys())}", flush=True)
            for strat in adapters.keys():
                obj = maybe_load_model(strat)
                _ = "ready" if obj is not None else "not-found"
                print(f"[MODEL-PRELOAD:{strat}] { _ }.", flush=True)
        except Exception as e:
            print(f"[BOOT] preload error: {e}", flush=True)

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
