import os, time, importlib
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List

import pandas as pd

from app.data_alpaca import fetch_alpaca_1m
from app.broker_alpaca import place_bracket_order, get_positions_symbols
from app.strategy_registry import ADAPTERS

# -------- ENV (tune from Render) ----------
SYMBOLS = [s.strip().upper() for s in os.getenv("SCANNER_SYMBOLS","SPY,QQQ").split(",") if s.strip()]
TF_MIN_LIST = [int(x) for x in os.getenv("TF_MIN_LIST","5,15").split(",") if x.strip()]
ACTIVE = [s.strip() for s in os.getenv("ACTIVE_STRATEGIES","trading_bot,ml_merged,ranked_ml").split(",") if s.strip()]
LOOKBACK_MIN = int(os.getenv("LOOKBACK_MIN","2400"))     # ~40h of 1m bars
POLL_SECONDS = int(os.getenv("POLL_SECONDS","20"))

DRY_RUN = os.getenv("DRY_RUN","1") in ("1","true","True")
DEDUP_BY_SYMBOL = os.getenv("DEDUP_BY_SYMBOL","1") in ("1","true","True")
SKIP_IF_POSITION_OPEN = os.getenv("SKIP_IF_POSITION_OPEN","1") in ("1","true","True")

# 🔑 Only these strategies are ranked. Default: ranked_ml only.
RANKED_STRATS = [s.strip() for s in os.getenv("RANKED_STRATS","ranked_ml").split(",") if s.strip()]

# Defaults / per-strategy overrides
TOPK_RANKED_DEFAULT = int(os.getenv("TOPK_RANKED_DEFAULT","1"))       # used if TOPK_<STRAT> not set (for ranked strats)
CONF_THR_DEFAULT = float(os.getenv("CONF_THR_DEFAULT","0.05"))
MIN_QTY_DEFAULT = int(os.getenv("MIN_QTY_DEFAULT","1"))
# For non-ranked strats, we don't cap unless TOPK_<STRAT> is set
_NONRANK_BIG = 10**9

def _per_strat(key: str, strat: str, cast, default):
    env_key = f"{key}_{strat.upper()}"
    v = os.getenv(env_key)
    return cast(v) if v is not None else default

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
        mods[name] = importlib.import_module(path)
    return mods

def _norm_qty(q, strat):
    try:
        q = int(q or 0)
    except Exception:
        q = 0
    return q if q > 0 else _per_strat("MIN_QTY", strat, int, MIN_QTY_DEFAULT)

def scan_once(adapters) -> List[Candidate]:
    candidates: List[Candidate] = []
    open_syms = get_positions_symbols() if SKIP_IF_POSITION_OPEN else set()

    for sym in SYMBOLS:
        df1m = fetch_alpaca_1m(sym, limit=LOOKBACK_MIN)
        if df1m is None or df1m.empty:
            continue
        last = float(df1m["close"].iloc[-1])

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
                    # simple long-only skip if already in a long; adjust if you run shorts
                    continue

                qty = _norm_qty(sig.get("quantity"), strat)
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

        # Sort by confidence (desc) so either mode is consistent
        items.sort(key=lambda x: x.confidence, reverse=True)

        if ranked_mode:
            # Rank-and-take-topK ONLY for ranked_ml (or whatever is in RANKED_STRATS)
            topk = _per_strat("TOPK", strat, int, TOPK_RANKED_DEFAULT)
            take = items[:max(0, topk)]
            print(f"[RANK] {strat}: {len(items)} candidates -> taking top {len(take)}", flush=True)
            orders.extend(take)
        else:
            # Non-ranked strategies: execute everything (unless TOPK_<STRAT> is set)
            topk = _per_strat("TOPK", strat, int, _NONRANK_BIG)
            take = items[:max(0, topk)]
            print(f"[PASS] {strat}: {len(items)} candidates -> passing {len(take)}", flush=True)
            orders.extend(take)

    # Execute (or dry-run)
    for o in orders:
        if DRY_RUN:
            print(f"[DRY] {o.strategy} conf={o.confidence:.4f} {o.symbol}:{o.tf}m side={o.side} qty={o.qty} tp={o.tp} sl={o.sl}", flush=True)
        else:
            ok, info = place_bracket_order(o.symbol, o.side, o.qty, o.last, o.tp, o.sl)
            print(f"[ORDER] {o.strategy} {o.symbol}:{o.tf}m conf={o.confidence:.4f} side={o.side} qty={o.qty} ok={ok} info={str(info)[:160]}", flush=True)

def main():
    adapters = _load_adapters()
    print(f"[BOOT] ACTIVE={list(adapters.keys())} SYMBOLS={SYMBOLS} TFs={TF_MIN_LIST} DRY_RUN={int(DRY_RUN)} RANKED_STRATS={RANKED_STRATS}", flush=True)

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
