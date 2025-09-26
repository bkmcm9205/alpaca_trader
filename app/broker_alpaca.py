# app/broker_alpaca.py
from __future__ import annotations
import os, time
from collections import deque
from typing import Dict, Tuple, Set, Any, Optional
from decimal import Decimal, ROUND_HALF_UP, getcontext
import requests

ALPACA_TRADING_BASE = os.getenv("ALPACA_TRADING_BASE", "https://paper-api.alpaca.markets").rstrip("/")
ALPACA_KEY          = os.getenv("ALPACA_KEY_ID", os.getenv("APCA_API_KEY_ID", ""))
ALPACA_SECRET       = os.getenv("ALPACA_SECRET_KEY", os.getenv("APCA_API_SECRET_KEY", ""))

ORDER_RATE_LIMIT_PM = int(os.getenv("ORDER_RATE_LIMIT_PM", "120"))
STRAT_TAG           = os.getenv("STRAT_TAG", "").strip()
USD_PER_TRADE       = float(os.getenv("USD_PER_TRADE", "0") or 0.0)

TICK_ABOVE_1 = os.getenv("TICK_ABOVE_1", "0.01")
TICK_BELOW_1 = os.getenv("TICK_BELOW_1", "0.0001")

getcontext().prec = 12

def _auth_headers() -> Dict[str,str]:
    if not ALPACA_KEY or not ALPACA_SECRET:
        raise RuntimeError("Missing Alpaca credentials (ALPACA_KEY_ID / ALPACA_SECRET_KEY).")
    return {
        "Apca-Api-Key-Id": ALPACA_KEY,
        "Apca-Api-Secret-Key": ALPACA_SECRET,
        "accept": "application/json",
        "content-type": "application/json",
    }

def _orders_url() -> str:     return f"{ALPACA_TRADING_BASE}/v2/orders"
def _positions_url() -> str:  return f"{ALPACA_TRADING_BASE}/v2/positions"
def _position_url(sym:str)->str: return f"{ALPACA_TRADING_BASE}/v2/positions/{sym.upper()}"
def _account_url() -> str:    return f"{ALPACA_TRADING_BASE}/v2/account"
def _asset_url(sym:str)->str: return f"{ALPACA_TRADING_BASE}/v2/assets/{sym.upper()}"

def _dec(x): return Decimal(str(x))
def _tick_for(last: float) -> Decimal:
    return _dec(TICK_ABOVE_1) if _dec(last) >= _dec("1.0") else _dec(TICK_BELOW_1)

def _snap(price: Optional[float]) -> Optional[float]:
    if price is None: return None
    d = _dec(price)
    tick = _tick_for(price)
    return float((d / tick).to_integral_value(rounding=ROUND_HALF_UP) * tick)

def _tp_from(last: float, tp: Optional[float], is_long: bool, extra_ticks:int=2) -> Optional[float]:
    if tp is None: return None
    v = float(tp)
    raw = (last * (1.0 + v)) if (0 < v < 1 and is_long) else \
          (last * (1.0 - v)) if (0 < v < 1 and not is_long) else v
    base = _snap(raw)
    # cushion: ensure ≥ last + 2*ticks for long; ≤ last - 2*ticks for short
    tick = float(_tick_for(last))
    if is_long and (base is None or base <= last + tick):
        base = _snap(last + tick * extra_ticks)
    if (not is_long) and (base is None or base >= last - tick):
        base = _snap(last - tick * extra_ticks)
    return base

def _sl_from(last: float, sl: Optional[float], is_long: bool) -> Optional[float]:
    if sl is None: return None
    v = float(sl)
    raw = (last * (1.0 - v)) if (0 < v < 1 and is_long) else \
          (last * (1.0 + v)) if (0 < v < 1 and not is_long) else v
    return _snap(raw)

def _positions_map() -> Dict[str, Dict[str, Any]]:
    try:
        r = requests.get(_positions_url(), headers=_auth_headers(), timeout=20)
        if r.status_code == 404: return {}
        r.raise_for_status()
        out = {}
        for p in r.json() or []:
            sym = str(p.get("symbol","")).upper()
            out[sym] = p
        return out
    except requests.RequestException:
        return {}

def _asset(sym: str) -> Dict[str, Any]:
    r = requests.get(_asset_url(sym), headers=_auth_headers(), timeout=15)
    r.raise_for_status()
    return r.json()

def get_account_summary() -> Dict[str, Any]:
    r = requests.get(_account_url(), headers=_auth_headers(), timeout=30)
    r.raise_for_status()
    return r.json()

def get_positions_symbols() -> Set[str]:
    return set(_positions_map().keys())

_order_bucket: deque[float] = deque()
def _rate_limit_block():
    now = time.time()
    while _order_bucket and now - _order_bucket[0] > 60.0: _order_bucket.popleft()
    if len(_order_bucket) >= ORDER_RATE_LIMIT_PM:
        time.sleep(max(0.05, 60.0 - (now - _order_bucket[0])))
    _order_bucket.append(time.time())

def place_bracket_order(
    symbol: str,
    side: str,                # "buy" or "sell"
    qty: int,
    last_price: float,
    tp: float | None,
    sl: float | None
) -> Tuple[bool, Dict[str, Any] | str]:
    if not symbol or not side:
        return False, "missing symbol or side"
    side = side.lower().strip()
    if side not in ("buy","sell"):
        return False, f"invalid side '{side}'"

    # ----- Optional fallback sizing if qty==0 and USD_PER_TRADE set
    if (qty or 0) <= 0 and USD_PER_TRADE > 0 and last_price > 0:
        qty = max(1, int(USD_PER_TRADE / float(last_price)))

    # ----- Cap by current buying power (prevents 403)
    try:
        acct = get_account_summary()
        bp = float(acct.get("buying_power") or acct.get("buying_power", 0.0))
    except Exception:
        bp = 0.0
    if bp > 0 and last_price > 0:
        max_bp_qty = int(bp // float(last_price))
        if max_bp_qty < 1:
            return False, {"status": 403, "body": {"message":"insufficient buying power", "buying_power": bp}}
        if qty > max_bp_qty:
            qty = max_bp_qty

    if (qty or 0) <= 0:
        return False, "qty must be > 0"

    # --- Preflight asset & shortability (prevents 403 on non-shortables)
    try:
        a = _asset(symbol)
        if not a.get("tradable", False):
            return False, {"status":403, "body":{"message":"asset not tradable"}}
    except requests.RequestException as e:
        return False, f"asset lookup failed: {e}"

    # Determine if this SELL is opening a short (no existing long)
    pos = _positions_map().get(symbol.upper())
    has_long = False
    if pos:
        try:
            side_pos = pos.get("side","").lower()
            has_long = (side_pos == "long" and float(pos.get("qty","0")) > 0)
        except Exception:
            has_long = False
    is_short_open = (side == "sell" and not has_long)
    if is_short_open and not a.get("shortable", False):
        return False, {"status":403, "body":{"message":"shorting not allowed (shortable=false)"}}

    is_long = (side == "buy")
    # Child prices with safety cushion (prevents 422 base_price +/- 0.01)
    tp_price = _tp_from(last_price, tp, is_long, extra_ticks=2)
    sl_price = _sl_from(last_price, sl, is_long)

    # Parent MARKET + bracket children (if any)
    tag = STRAT_TAG or "STRAT"
    client_order_id = f"{tag}-{symbol}-{int(time.time())}"
    data: Dict[str, Any] = {
        "symbol": symbol.upper(),
        "side": side,
        "type": "market",
        "time_in_force": "day",
        "qty": int(qty),
        "client_order_id": client_order_id,
    }
    if tp_price is not None or sl_price is not None:
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
            try: body = r.json()
            except Exception: body = r.text
            return False, {"status": r.status_code, "body": body, "payload": data}
        return True, r.json()
    except requests.RequestException as e:
        return False, str(e)
