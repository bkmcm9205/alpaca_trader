# app/broker_alpaca.py
from __future__ import annotations
import os, time, hashlib
from collections import deque
from typing import Dict, Tuple, Set, Any, List

import requests
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

# ====== Config (env) ======
ALPACA_TRADING_BASE = os.getenv("ALPACA_TRADING_BASE", "https://paper-api.alpaca.markets").rstrip("/")
# Back-compat: allow either ALPACA_* or APCA_* names
ALPACA_KEY          = os.getenv("ALPACA_KEY_ID", os.getenv("APCA_API_KEY_ID", ""))
ALPACA_SECRET       = os.getenv("ALPACA_SECRET_KEY", os.getenv("APCA_API_SECRET_KEY", ""))

# Per-minute order rate limit (simple in-process limiter)
ORDER_RATE_LIMIT_PM = int(os.getenv("ORDER_RATE_LIMIT_PM", "120"))

# Strategy tagging (for client_order_id)
STRAT_TAG     = os.getenv("STRAT_TAG", "").strip()             # e.g., TB / MM / RML / MS
STRATEGY_ID   = os.getenv("STRATEGY_ID", STRAT_TAG).strip() or "STRAT"
# ACTIVE_STRATEGIES can be comma-separated; we use the first token as the strategy name
_active_strats = [s.strip() for s in os.getenv("ACTIVE_STRATEGIES", "").split(",") if s.strip()]
STRATEGY_NAME = _active_strats[0] if _active_strats else "unknown"

# Optional sizing helper if a signal gives qty 0 (kept for back-compat; can leave 0)
USD_PER_TRADE = float(os.getenv("USD_PER_TRADE", "0") or 0.0)

# Market timezone for date scoping (activities fetch)
MARKET_TZ = os.getenv("MARKET_TZ", "America/New_York")

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

def _activities_url() -> str:
    return f"{ALPACA_TRADING_BASE}/v2/account/activities"

def _normalize_price_from_pct_or_abs(last: float, val: float | None, upward: bool) -> float | None:
    """
    Interpret TP/SL as:
      - None -> None
      - 0 < val < 1 -> percentage offset from `last`
      - val >= 1 -> absolute price
    `upward=True` for take-profit (above last), False for stop (below last).
    """
    if val is None:
        return None
    try:
        v = float(val)
    except Exception:
        return None
    if v <= 0:
        return None
    if v < 1.0:
        # percent
        return round(last * (1.0 + v if upward else 1.0 - v), 4)
    # absolute
    return round(v, 4)

# very small, local token bucket to avoid burst rejections
_order_bucket: deque[float] = deque()

def _rate_limit_block():
    now = time.time()
    # drop entries older than 60s
    while _order_bucket and now - _order_bucket[0] > 60.0:
        _order_bucket.popleft()
    if len(_order_bucket) >= ORDER_RATE_LIMIT_PM:
        # wait until the oldest request falls out of the 60s window
        sleep_for = 60.0 - (now - _order_bucket[0])
        time.sleep(max(0.05, sleep_for))
    _order_bucket.append(time.time())

def _make_client_order_id(symbol: str) -> str:
    """
    Build a stable, short client_order_id with strategy attribution.
    Format (<=48 chars): <STRATEGY_ID>-<STRATEGY_NAME>-<YYYYMMDD-HHMMSS>-<4hex>
    """
    ts = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
    seed = f"{STRATEGY_ID}-{STRATEGY_NAME}-{symbol}-{ts}"
    nonce = hashlib.sha1(seed.encode()).hexdigest()[:4]
    coid = f"{STRATEGY_ID}-{STRATEGY_NAME}-{ts}-{nonce}"
    return coid[:48]

# ====== Public API used by strategy_runner ======
def get_account_summary() -> Dict[str, Any]:
    r = requests.get(_account_url(), headers=_auth_headers(), timeout=30)
    r.raise_for_status()
    return r.json()

def get_positions_symbols() -> Set[str]:
    """
    Returns the set of symbols that currently have an open position (long or short).
    """
    try:
        r = requests.get(_positions_url(), headers=_auth_headers(), timeout=30)
        # Alpaca returns 200 with [] if no positions
        if r.status_code == 404:
            return set()
        r.raise_for_status()
        arr = r.json() or []
        syms = {str(p.get("symbol", "")).upper() for p in arr if p.get("symbol")}
        return syms
    except requests.RequestException:
        return set()

def cancel_all_orders() -> Tuple[bool, Any]:
    """
    Cancel all open orders.
    DELETE /v2/orders
    """
    try:
        r = requests.delete(_orders_url(), headers=_auth_headers(), timeout=30)
        # 204 No Content on success; some clients return 207-299; accept 2xx
        if 200 <= r.status_code < 300:
            return True, r.text or "{}"
        return False, f"{r.status_code} {r.text}"
    except requests.RequestException as e:
        return False, str(e)

def close_all_positions() -> Tuple[bool, Any]:
    """
    Close all open positions.
    DELETE /v2/positions
    """
    try:
        r = requests.delete(_positions_url(), headers=_auth_headers(), timeout=60)
        if 200 <= r.status_code < 300:
            return True, r.text or "{}"
        return False, f"{r.status_code} {r.text}"
    except requests.RequestException as e:
        return False, str(e)

def place_bracket_order(
    symbol: str,
    side: str,                # "buy" or "sell"
    qty: int,
    last_price: float,
    tp: float | None,         # percent (0-1) or absolute price
    sl: float | None          # percent (0-1) or absolute price
) -> Tuple[bool, Dict[str, Any] | str]:
    """
    Places a MARKET order; if tp and/or sl are provided, uses a 'bracket' with take_profit/stop_loss.
    Returns (ok, info) where ok=True if 2xx and info is JSON; otherwise ok=False with error text.
    """
    if not symbol or not side:
        return False, "missing symbol or side"

    side = side.lower().strip()
    if side not in ("buy", "sell"):
        return False, f"invalid side '{side}'"

    # Fallback sizing: if qty <= 0 and USD_PER_TRADE is set, derive qty from last_price
    if (qty or 0) <= 0 and USD_PER_TRADE > 0 and (last_price or 0) > 0:
        qty = max(1, int(float(USD_PER_TRADE) / float(last_price)))

    if (qty or 0) <= 0:
        return False, "qty must be > 0"

    # Normalize TP/SL
    tp_price = _normalize_price_from_pct_or_abs(last_price, tp, upward=True)
    sl_price = _normalize_price_from_pct_or_abs(last_price, sl, upward=False)

    # Strategy-aware client order id
    client_order_id = _make_client_order_id(symbol)

    data: Dict[str, Any] = {
        "symbol": symbol.upper(),
        "side": side,
        "type": "market",
        "time_in_force": "day",
        "qty": int(qty),
        "client_order_id": client_order_id,
    }

    if tp_price or sl_price:
        data["order_class"] = "bracket"
        if tp_price:
            data["take_profit"] = {"limit_price": float(tp_price)}
        if sl_price:
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

# ---------- Fills (for per-strategy ledgers / S3 trade logs) ----------
def _today_et_date_str() -> str:
    # Activities API 'date' param is in YYYY-MM-DD (session date). Use ET.
    return datetime.now(ZoneInfo(MARKET_TZ)).date().isoformat()

def get_today_fills() -> List[Dict[str, Any]]:
    """
    Return today's fills from Alpaca Activities API (no args).
    GET /v2/account/activities?activity_types=FILL&date=YYYY-MM-DD
    Handles pagination via X-Next-Page-Token header.
    """
    params = {"activity_types": "FILL", "date": _today_et_date_str(), "page_size": 100}
    url = _activities_url()
    out: List[Dict[str, Any]] = []
    try:
        while True:
            r = requests.get(url, headers=_auth_headers(), params=params, timeout=30)
            if r.status_code == 404:
                # no activities for the day
                return out
            r.raise_for_status()
            page = r.json() or []
            if page:
                out.extend(page)
            token = r.headers.get("X-Next-Page-Token") or r.headers.get("x-next-page-token")
            if not token:
                break
            params["page_token"] = token
    except requests.RequestException as e:
        # Return what we have (possibly empty) and let caller log/handle
        return out
    return out
