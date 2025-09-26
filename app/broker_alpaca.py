from __future__ import annotations
import os, time
from collections import deque
from typing import Dict, Tuple, Set, Any
from decimal import Decimal, ROUND_HALF_UP, getcontext

import requests

# ====== Config (env) ======
ALPACA_TRADING_BASE = os.getenv("ALPACA_TRADING_BASE", "https://paper-api.alpaca.markets").rstrip("/")
ALPACA_KEY          = os.getenv("ALPACA_KEY_ID", os.getenv("APCA_API_KEY_ID", ""))
ALPACA_SECRET       = os.getenv("ALPACA_SECRET_KEY", os.getenv("APCA_API_SECRET_KEY", ""))

# Per-minute order rate limit (simple in-process limiter)
ORDER_RATE_LIMIT_PM = int(os.getenv("ORDER_RATE_LIMIT_PM", "120"))

# Optional per-strategy tag for traceability
STRAT_TAG = os.getenv("STRAT_TAG", "").strip()  # e.g., TB / MM / RML

# Optional sizing helper if a signal gives qty 0
USD_PER_TRADE = float(os.getenv("USD_PER_TRADE", "0") or 0.0)

# Tick sizes (overridable)
TICK_ABOVE_1  = os.getenv("TICK_ABOVE_1",  "0.01")
TICK_BELOW_1  = os.getenv("TICK_BELOW_1",  "0.0001")

getcontext().prec = 12  # sufficient precision for equity ticks

# ====== Internal helpers ======
def _auth_headers() -> Dict[str, str]:
    if not ALPACA_KEY or not ALPACA_SECRET:
        raise RuntimeError("Missing Alpaca credentials (ALPACA_KEY_ID / ALPACA_SECRET_KEY).")
    return {
        "Apca-Api-Key-Id": ALPACA_KEY,
        "Apca-Api-Secret-Key": ALPACA_SECRET,
        "accept": "application/json",
        "content-type": "application/json",
    }

def _orders_url() -> str:
    return f"{ALPACA_TRADING_BASE}/v2/orders"

def _positions_url() -> str:
    return f"{ALPACA_TRADING_BASE}/v2/positions"

def _account_url() -> str:
    return f"{ALPACA_TRADING_BASE}/v2/account"

def _decimal(x: float | str) -> Decimal:
    return Decimal(str(x))

def _snap_to_equity_tick(price: float | None) -> float | None:
    """
    Snap to the appropriate equity tick:
      - >= $1.00 -> $0.01
      -  < $1.00 -> $0.0001
    Overridable via TICK_ABOVE_1 / TICK_BELOW_1.
    """
    if price is None:
        return None
    d = _decimal(price)
    tick = _decimal(TICK_ABOVE_1) if d >= _decimal("1.0") else _decimal(TICK_BELOW_1)
    snapped = (d / tick).to_integral_value(rounding=ROUND_HALF_UP) * tick
    return float(snapped)

def _normalize_price_from_pct_or_abs(last: float, val: float | None, upward: bool) -> float | None:
    """
    Interpret TP/SL values:
      - None -> None
      - 0 < val < 1 -> treat as % offset from `last` (upward=True for TP, False for SL)
      - val >= 1 -> treat as absolute price
    Always returns a tick-snapped float or None.
    """
    if val is None:
        return None
    try:
        v = float(val)
    except Exception:
        return None
    if v <= 0:
        return None
    raw = (last * (1.0 + v)) if (v < 1.0 and upward) else \
          (last * (1.0 - v)) if (v < 1.0 and not upward) else \
          v
    return _snap_to_equity_tick(raw)

# tiny token bucket to avoid burst rate errors
_order_bucket: deque[float] = deque()

def _rate_limit_block():
    now = time.time()
    while _order_bucket and now - _order_bucket[0] > 60.0:
        _order_bucket.popleft()
    if len(_order_bucket) >= ORDER_RATE_LIMIT_PM:
        sleep_for = 60.0 - (now - _order_bucket[0])
        time.sleep(max(0.05, sleep_for))
    _order_bucket.append(time.time())

# ====== Public API used by strategy_runner ======
def get_account_summary() -> Dict[str, Any]:
    r = requests.get(_account_url(), headers=_auth_headers(), timeout=30)
    r.raise_for_status()
    return r.json()

def get_positions_symbols() -> Set[str]:
    """
    Returns symbols with an open position.
    """
    try:
        r = requests.get(_positions_url(), headers=_auth_headers(), timeout=30)
        if r.status_code == 404:
            return set()
        r.raise_for_status()
        arr = r.json() or []
        return {str(p.get("symbol", "")).upper() for p in arr if p.get("symbol")}
    except requests.RequestException:
        return set()

def place_bracket_order(
    symbol: str,
    side: str,                # "buy" or "sell"
    qty: int,
    last_price: float,
    tp: float | None,         # percent (0-1) or absolute price
    sl: float | None          # percent (0-1) or absolute price
) -> Tuple[bool, Dict[str, Any] | str]:
    """
    Places a MARKET parent order. If tp/sl provided, attaches Alpaca 'bracket' children:
      - take_profit.limit_price (tick-snapped)
      - stop_loss.stop_price    (tick-snapped)
    Returns (ok, info) where ok=True if 2xx and info is JSON; else ok=False with error text.
    """
    if not symbol or not side:
        return False, "missing symbol or side"

    side = side.lower().strip()
    if side not in ("buy", "sell"):
        return False, f"invalid side '{side}'"

    # Fallback sizing: if qty <= 0 and USD_PER_TRADE is set, derive qty from last_price
    if (qty or 0) <= 0 and USD_PER_TRADE > 0 and last_price > 0:
        qty = max(1, int(USD_PER_TRADE / float(last_price)))

    if (qty or 0) <= 0:
        return False, "qty must be > 0"

    # Compute tick-snapped TP/SL
    tp_price = _normalize_price_from_pct_or_abs(last_price, tp, upward=True)
    sl_price = _normalize_price_from_pct_or_abs(last_price, sl, upward=False)

    # Build MARKET parent
    tag = STRAT_TAG or "STRAT"
    client_order_id = f"{tag}-{symbol}-{int(time.time())}"
    data: Dict[str, Any] = {
        "symbol": symbol.upper(),
        "side": side,
        "type": "market",      # parent is always market
        "time_in_force": "day",
        "qty": int(qty),
        "client_order_id": client_order_id,
    }

    # Attach bracket children only if we have at least one valid price
    if (tp_price is not None) or (sl_price is not None):
        data["order_class"] = "bracket"
        if tp_price is not None:
            data["take_profit"] = {"limit_price": float(tp_price)}
        if sl_price is not None:
            data["stop_loss"] = {"stop_price": float(sl_price)}

    try:
        _rate_limit_block()
        r = requests.post(_orders_url(), headers=_auth_headers(), json=data, timeout=30)
        ok = 200 <= r.status_code < 300
        if not ok:
            try:
                body = r.json()
            except Exception:
                body = r.text
            return False, f"{r.status_code} {body}"
        return True, r.json()
    except requests.RequestException as e:
        return False, str(e)
