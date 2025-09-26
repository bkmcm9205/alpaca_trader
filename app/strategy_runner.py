from __future__ import annotations
import os, time, importlib, hashlib
from dataclasses import dataclass
from typing import Dict, List, Optional
from collections import defaultdict

import pandas as pd
from datetime import datetime
from zoneinfo import ZoneInfo

from app.data_alpaca import fetch_alpaca_1m
from app.broker_alpaca import (
    place_bracket_order,
    get_positions_symbols,
)

# Optional broker funcs (equity, cancel/close, fills)
try:
    from app.broker_alpaca import get_account_summary
except Exception:
    get_account_summary = None  # type: ignore

try:
    from app.broker_alpaca import cancel_all_orders, close_all_positions
except Exception:
    cancel_all_orders = None  # type: ignore
    close_all_positions = None  # type: ignore

try:
    from app.broker_alpaca import get_today_fills  # should return list of dicts
except Exception:
    get_today_fills = None  # type: ignore

# Per-strategy logging + virtual equity
_PNL_OK = False
try:
    from app.pnl_logger import append_trades, update_ledger_with_fills, compute_virtual_equity  # type: ignore
    _PNL_OK = True
except Exception:
    _PNL_OK = False

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
STRATEGY_NAME = ACTIVE[0] if ACTIVE else "unknown"  # first active name (your runners use one strategy per worker)

LOOKBACK_MIN = int(os.getenv("LOOKBACK_MIN","2400"))     # ~40h of 1m bars
POLL_SECONDS = int(os.getenv("POLL_SECONDS","20"))

DRY_RUN = os.getenv("DRY_RUN","1").lower() in ("1","true","yes")
DEDUP_BY_SYMBOL = os.getenv("DEDUP_BY_SYMBOL","1").lower() in ("1","true","yes")
SKIP_IF_POSITION_OPEN = os.getenv("SKIP_IF_POSITION_OPEN","1").lower() in ("1","true","yes")

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

# ===== Guard + logging ENV =====
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

        if DEDUP_BY_SYMBOL:
            best_by_symbol: Dict[str, Candidate] = {}
            for c in items:
                cur = best_by_symbol.get(c.symbol)
                if cur is None or c.confidence > cur.confidence:
                    best_by_symbol[c.symbol] = c
            items = list(best_by_symbol.values())

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
        if isinstance(acct, dict) and "equity" in acct:
            return float(acct["equity"])
    except Exception as e:
        print(f"[GUARD] account equity fetch error: {e}", flush=True)
    return None

def _maybe_pull_and_log_fills(eq_now: Optional[float]) -> Dict[str, float]:
    """
    Pull today's fills, filter by our client_order_id prefix STRATEGY_ID-STRATEGY_NAME,
    update per-strategy ledger, and append S3 trade rows.
    Returns a price_map (symbol->last price) derived from observed fills; the scan will enrich more.
    """
    price_map: Dict[str, float] = {}
    if not (_PNL_OK and callable(get_today_fills)):  # type: ignore
        return price_map

    try:
        fills = get_today_fills()  # expected to return list of dicts
    except TypeError:
        # If your get_today_fills requires args, wire a no-arg wrapper in broker_alpaca.
        try:
            fills = []  # graceful fallback
        except Exception:
            fills = []
    except Exception:
        fills = []

    if not fills:
        return price_map

    prefix = f"{STRATEGY_ID}-{STRATEGY_NAME}"
    my_fills = [f for f in fills if str(f.get("client_order_id","")).startswith(prefix)]
    if not my_fills:
        return price_map

    # Update per-strategy ledger with our fills
    try:
        update_ledger_with_fills(STRATEGY_ID, [
            {
                "symbol": f.get("symbol"),
                "side": str(f.get("side","")).lower(),
                "qty":  f.get("qty", f.get("quantity", "0")),
                "price":f.get("price", f.get("fill_price", "0")),
            } for f in my_fills if f.get("symbol")
        ])
    except Exception as e:
        print(f"[PNL] ledger update error: {e}", flush=True)

    # Append S3 trade log rows
    try:
        ts = datetime.now().astimezone().isoformat()
        rows = []
        for f in my_fills:
            sym = f.get("symbol")
            if sym:
                try:
                    price_map[sym] = float(f.get("price", f.get("fill_price", "0")) or 0.0)
                except Exception:
                    pass
            rows.append({
                "ts_utc": ts,
                "symbol": sym or "",
                "side": str(f.get("side","")).lower(),
                "qty": str(f.get("qty", f.get("quantity",""))),
                "avg_fill": str(f.get("price", f.get("fill_price",""))),
                "client_order_id": str(f.get("client_order_id","")),
                "strategy": STRATEGY_NAME,
                "tag": STRATEGY_ID,
                "realized_pnl": str(f.get("profit_loss","")),
                "unrealized_pnl": "",
                "account_equity": "" if eq_now is None else f"{eq_now}",
            })
        append_trades(rows)
    except Exception as e:
        print(f"[PNL] append trades error: {e}", flush=True)

    return price_map

def _maybe_guard_and_halt(eq_now: float, baseline: float) -> bool:
    """Returns True if guard tripped (and performs flatten if configured)."""
    global _halt_entries
    tp_hit = eq_now >= baseline * (1.0 + DAILY_PROFIT_TARGET_PCT)
    dd_hit = eq_now <= baseline * (1.0 - DAILY_DRAWDOWN_PCT)
    if tp_hit or dd_hit:
        _guard_log(STRATEGY_ID, baseline, eq_now, DAILY_PROFIT_TARGET_PCT, DAILY_DRAWDOWN_PCT, star=True)
        _halt_entries = True
        if GUARD_FLATTEN and not DRY_RUN:
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

    _account_baseline = None
    _halt_entries = False

    while True:
        loop_start = time.time()
        try:
            # ===== Equity baseline (post-open) =====
            eq_acct = _account_equity_now()
            if _account_baseline is None and _market_opened_et() and eq_acct is not None:
                _account_baseline = float(eq_acct)
                print(f"[GUARD:{STRATEGY_ID}] session baseline set to {_fmt_usd(_account_baseline)}.", flush=True)

            # ===== Pull fills → ledger + S3 trades =====
            price_map_from_fills = _maybe_pull_and_log_fills(eq_acct)

            # ===== Scan & rank =====
            cands = scan_once(adapters)

            # Build price_map from candidates' latest prices (augment)
            price_map: Dict[str, float] = dict(price_map_from_fills)
            for c in cands:
                if c.symbol not in price_map:
                    price_map[c.symbol] = c.last

            # ===== Determine equity now (virtual if available) =====
            eq_now: Optional[float] = None
            baseline_for_log: Optional[float] = _account_baseline

            if USE_VIRTUAL_EQUITY and _PNL_OK and (_account_baseline is not None):
                try:
                    veq = compute_virtual_equity(STRATEGY_ID, _account_baseline, price_map)
                    if isinstance(veq, dict) and "virtual_equity" in veq:
                        eq_now = float(veq["virtual_equity"])
                        baseline_for_log = float(veq.get("baseline", _account_baseline))
                except Exception as e:
                    print(f"[GUARD] virtual equity error: {e}", flush=True)

            if eq_now is None and eq_acct is not None:
                eq_now = float(eq_acct)

            # ===== Guard logs + checks =====
            if (eq_now is not None) and (baseline_for_log is not None):
                if _GUARD_LOOP_COUNTER == 0:
                    _guard_log(STRATEGY_ID, baseline_for_log, eq_now, DAILY_PROFIT_TARGET_PCT, DAILY_DRAWDOWN_PCT, star=False)
                else:
                    if _GUARD_DEBUG or (_GUARD_LOOP_COUNTER % max(1, _GUARD_EVERY_N) == 0):
                        _guard_log(STRATEGY_ID, baseline_for_log, eq_now, DAILY_PROFIT_TARGET_PCT, DAILY_DRAWDOWN_PCT, star=False)

                if not _halt_entries and _maybe_guard_and_halt(eq_now, baseline_for_log):
                    pass  # guard tripped; entries halted

            _GUARD_LOOP_COUNTER += 1

            # ===== EOD flatten window =====
            if _within_eod_flatten_window():
                if not DRY_RUN and close_all_positions:
                    try:
                        print("[EOD] Auto-flatten window.", flush=True)
                        if cancel_all_orders: cancel_all_orders()
                        close_all_positions()
                    except Exception as e:
                        print(f"[EOD] flatten error: {e}", flush=True)

            # ===== Execute =====
            orders = rank_and_select(cands)
            execute(orders)

        except Exception as e:
            print(f"[LOOP ERROR] {e}", flush=True)

        elapsed = time.time() - loop_start
        time.sleep(max(1, POLL_SECONDS - int(elapsed)))

if __name__ == "__main__":
    main()
