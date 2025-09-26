from __future__ import annotations
import os, time, importlib
from dataclasses import dataclass
from typing import Dict, List, Optional
from collections import defaultdict

import pandas as pd
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from app.data_alpaca import fetch_alpaca_1m
from app.broker_alpaca import (
    place_bracket_order,
    get_positions_symbols,
)

# Optional broker funcs (guard/EOD actions, equity)
try:
    from app.broker_alpaca import get_account_summary
except Exception:  # pragma: no cover
    get_account_summary = None  # type: ignore

try:
    from app.broker_alpaca import cancel_all_orders, close_all_positions
except Exception:  # pragma: no cover
    cancel_all_orders = None  # type: ignore
    close_all_positions = None  # type: ignore

# Optional per-strategy ledger/virtual equity support
_PNL_LOGGER_OK = False
try:
    from app.pnl_logger import compute_virtual_equity  # type: ignore
    _PNL_LOGGER_OK = True
except Exception:
    _PNL_LOGGER_OK = False

from app.strategy_registry import ADAPTERS

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
MIN_QTY_DEFAULT = int(os.getenv("MIN_QTY_DEFAULT","1"))
_NONRANK_BIG = 10**9  # effectively "no cap" for non-ranked unless TOPK_<STRAT> set

def _per_strat(key: str, strat: str, cast, default):
    env_key = f"{key}_{strat.upper()}"
    v = os.getenv(env_key)
    return cast(v) if v is not None else default

# ===== Guard ENV =====
STRATEGY_ID = os.getenv("STRATEGY_ID", os.getenv("STRAT_TAG","GEN")).strip() or "GEN"
USE_VIRTUAL_EQUITY = os.getenv("USE_VIRTUAL_EQUITY","0").lower() in ("1","true","yes")
DAILY_PROFIT_TARGET_PCT = float(os.getenv("DAILY_PROFIT_TARGET_PCT","0.10"))
DAILY_DRAWDOWN_PCT      = float(os.getenv("DAILY_DRAWDOWN_PCT","0.05"))
GUARD_FLATTEN           = os.getenv("GUARD_FLATTEN","true").lower() in ("1","true","yes")
# EOD flatten window (ET)
EOD_FLATTEN_START_ET    = os.getenv("EOD_FLATTEN_START_ET","16:00")
EOD_FLATTEN_END_ET      = os.getenv("EOD_FLATTEN_END_ET","16:10")
# Guard logging verbosity
_GUARD_DEBUG            = os.getenv("GUARD_DEBUG","0").lower() in ("1","true","yes")
_GUARD_EVERY_N          = int(os.getenv("GUARD_LOG_EVERY_N_LOOPS","12"))

# =============================
#   PRE-MARKET UNIVERSE LOADER
# =============================

_universe_syms: List[str] = []
_universe_loaded_day = None
_universe_last_check = 0.0

def _now_et() -> datetime:
    return datetime.now(ZoneInfo("America/New_York"))

def _in_window(window: str) -> bool:
    try:
        s,e = window.split("-",1)
        now_hm = _now_et().strftime("%H:%M")
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
        today = _now_et().date()
        if _universe_loaded_day == today:
            return

    try:
        import boto3
        from botocore.exceptions import ClientError
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
            _universe_loaded_day = _now_et().date()
            print(f"[UNIVERSE] loaded {len(syms)} symbols from s3://{S3_BUCKET}/{UNIVERSE_S3_KEY}", flush=True)
    except Exception as e:
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
        mods[name] = importlib.import_module(path)
    return mods

def _norm_qty(q, strat):
    try:
        q = int(q or 0)
    except Exception:
        q = 0
    return q if q > 0 else _per_strat("MIN_QTY", strat, int, MIN_QTY_DEFAULT)

# =============================
#      GUARD LOGGING HELPERS
# =============================

def _fmt_usd(x) -> str:
    try:
        return f"${float(x):,.2f}"
    except Exception:
        return str(x)

_GUARD_LOOP_COUNTER = 0
_account_baseline: Optional[float] = None
_halt_entries: bool = False  # set when TP/DD hits

def _guard_log(strategy_id: str, baseline: float, eq_now: float,
               tp_pct: float, dd_pct: float, star: bool = False) -> None:
    up_lim = baseline * (1.0 + tp_pct)
    dn_lim = baseline * (1.0 - dd_pct)
    tag = "GUARD*" if star else "GUARD"
    print(
        f"[{tag}:{strategy_id}] baseline={_fmt_usd(baseline)}  "
        f"now={_fmt_usd(eq_now)}  "
        f"targets +{tp_pct*100:.1f}%({_fmt_usd(up_lim)}) / -{dd_pct*100:.1f}%({_fmt_usd(dn_lim)})",
        flush=True
    )

def _market_opened_et() -> bool:
    now = _now_et()
    return (now.hour > 9) or (now.hour == 9 and now.minute >= 30)

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

def rank_and_select(candidates: List[Candidate]) -> List[Candidate]:
    if not candidates:
        print("[RANK] no candidates", flush=True)
        return []

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

    return orders

def execute(orders: List[Candidate]):
    if not orders:
        return
    if _halt_entries:
        print(f"[GUARD:{STRATEGY_ID}] entries halted for the day.", flush=True)
        return

    for o in orders:
        if DRY_RUN:
            print(f"[DRY] {o.strategy} conf={o.confidence:.4f} {o.symbol}:{o.tf}m side={o.side} qty={o.qty} tp={o.tp} sl={o.sl}", flush=True)
        else:
            ok, info = place_bracket_order(o.symbol, o.side, o.qty, o.last, o.tp, o.sl)
            print(f"[ORDER] {o.strategy} {o.symbol}:{o.tf}m conf={o.confidence:.4f} side={o.side} qty={o.qty} ok={ok} info={str(info)[:160]}", flush=True)

# =============================
#       EQUITY & GUARDS
# =============================

def _account_equity_now() -> Optional[float]:
    if get_account_summary is None:
        return None
    try:
        acct = get_account_summary()
        # prefer "equity" if provided as float
        if isinstance(acct, dict) and "equity" in acct:
            return float(acct["equity"])
    except Exception as e:
        print(f"[GUARD] account equity fetch error: {e}", flush=True)
    return None

def _virtual_equity_now(price_map: Dict[str, float], baseline_hint: Optional[float]) -> Optional[float]:
    if not (USE_VIRTUAL_EQUITY and _PNL_LOGGER_OK):
        return None
    try:
        bh = float(baseline_hint) if (baseline_hint is not None) else None
    except Exception:
        bh = None
    try:
        # compute_virtual_equity returns dict with baseline, mtm, virtual_equity
        res = compute_virtual_equity(STRATEGY_ID, bh if bh is not None else 0.0, price_map)
        return float(res.get("virtual_equity"))
    except Exception as e:
        print(f"[GUARD] virtual equity error: {e}", flush=True)
        return None

def _maybe_guard_and_halt(eq_now: float, baseline: float) -> bool:
    """Returns True if guard tripped (and performs flatten if configured)."""
    global _halt_entries
    tp_hit = eq_now >= baseline * (1.0 + DAILY_PROFIT_TARGET_PCT)
    dd_hit = eq_now <= baseline * (1.0 - DAILY_DRAWDOWN_PCT)
    if tp_hit or dd_hit:
        _guard_log(STRATEGY_ID, baseline, eq_now, DAILY_PROFIT_TARGET_PCT, DAILY_DRAWDOWN_PCT, star=True)
        _halt_entries = True
        if GUARD_FLATTEN and not DRY_RUN:
            # Best-effort account-wide flatten (safer). If you later want per-strategy-only,
            # wire ledger-driven offsets here.
            try:
                if cancel_all_orders: cancel_all_orders()
                if close_all_positions: close_all_positions()
                print(f"[GUARD:{STRATEGY_ID}] FLATTEN requested (cancel orders + close positions).", flush=True)
            except Exception as e:
                print(f"[GUARD] flatten error: {e}", flush=True)
        else:
            print(f"[GUARD:{STRATEGY_ID}] Entries halted (no flatten in DRY_RUN={int(DRY_RUN)}).", flush=True)
        return True
    return False

def _within_eod_flatten_window() -> bool:
    try:
        start_h, start_m = [int(x) for x in EOD_FLATTEN_START_ET.split(":")]
        end_h, end_m     = [int(x) for x in EOD_FLATTEN_END_ET.split(":")]
        now = _now_et()
        start = now.replace(hour=start_h, minute=start_m, second=0, microsecond=0)
        end   = now.replace(hour=end_h, minute=end_m, second=0, microsecond=0)
        return start <= now <= end
    except Exception:
        return False

# =============================
#             MAIN
# =============================

def main():
    global _GUARD_LOOP_COUNTER, _account_baseline, _halt_entries

    adapters = _load_adapters()
    print(f"[BOOT] ACTIVE={list(adapters.keys())} TFs={TF_MIN_LIST} DRY_RUN={int(DRY_RUN)} RANKED_STRATS={RANKED_STRATS}", flush=True)
    if UNIVERSE_S3_KEY:
        print(f"[BOOT] Using S3 universe: s3://{S3_BUCKET}/{UNIVERSE_S3_KEY} (reload window {UNIVERSE_RELOAD_WINDOW_ET})", flush=True)
    elif SYMBOLS:
        print(f"[BOOT] Using SCANNER_SYMBOLS={SYMBOLS}", flush=True)
    else:
        print("[BOOT] No universe set; fallback to ['SPY','QQQ']", flush=True)

    # Attempt to set session baseline as soon as market opens and equity is available
    _account_baseline = None
    _halt_entries = False

    while True:
        loop_start = time.time()
        try:
            # =======================
            # Equity / Guard handling
            # =======================
            eq_acct = _account_equity_now()
            if _account_baseline is None and _market_opened_et() and eq_acct is not None:
                _account_baseline = float(eq_acct)
                print(f"[GUARD:{STRATEGY_ID}] session baseline set to {_fmt_usd(_account_baseline)}.", flush=True)

            # Build a light price map from last known prices (from quick sample of universe)
            price_map: Dict[str, float] = {}

            # Scan + rank (but only execute if not halted)
            cands = scan_once(adapters)

            # Use candidates’ last prices to enrich price_map (helps virtual equity MTM)
            for c in cands:
                price_map[c.symbol] = c.last

            # Determine equity now (virtual if enabled and available, else account)
            eq_now: Optional[float] = None
            baseline_for_log: Optional[float] = _account_baseline

            if USE_VIRTUAL_EQUITY and _PNL_LOGGER_OK and (_account_baseline is not None):
                veq = _virtual_equity_now(price_map, _account_baseline)
                if isinstance(veq, (int, float)):
                    eq_now = float(veq)
                    baseline_for_log = float(_account_baseline)
            if eq_now is None and eq_acct is not None:
                eq_now = float(eq_acct)

            # Periodic guard banner
            if (eq_now is not None) and (baseline_for_log is not None):
                if _GUARD_LOOP_COUNTER == 0:
                    _guard_log(STRATEGY_ID, baseline_for_log, eq_now, DAILY_PROFIT_TARGET_PCT, DAILY_DRAWDOWN_PCT, star=False)
                else:
                    if _GUARD_DEBUG or (_GUARD_LOOP_COUNTER % max(1, _GUARD_EVERY_N) == 0):
                        _guard_log(STRATEGY_ID, baseline_for_log, eq_now, DAILY_PROFIT_TARGET_PCT, DAILY_DRAWDOWN_PCT, star=False)

                # Check guard trip
                if not _halt_entries and _maybe_guard_and_halt(eq_now, baseline_for_log):
                    # if tripped, skip execution this loop (entries halted)
                    pass

            _GUARD_LOOP_COUNTER += 1

            # EOD flatten window (best-effort, idempotent)
            if _within_eod_flatten_window():
                if not DRY_RUN and close_all_positions:
                    try:
                        print("[EOD] Auto-flatten window.", flush=True)
                        if cancel_all_orders: cancel_all_orders()
                        close_all_positions()
                    except Exception as e:
                        print(f"[EOD] flatten error: {e}", flush=True)

            # Rank & possibly execute (if not halted)
            orders = rank_and_select(cands)
            execute(orders)

        except Exception as e:
            print(f"[LOOP ERROR] {e}", flush=True)

        elapsed = time.time() - loop_start
        time.sleep(max(1, POLL_SECONDS - int(elapsed)))

if __name__ == "__main__":
    main()
