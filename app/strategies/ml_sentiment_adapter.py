# app/strategies/ml_sentiment_adapter.py
from __future__ import annotations
import os
from typing import Optional, Dict, Any

import pandas as pd
import numpy as np

from app.strategies.model_runtime import (
    maybe_load_model, resample_tf, latest_features, proba_up_for
)

STRAT = "ml_sentiment"

# --- thresholds & risk knobs (per-strategy overrides supported) ---
CONF_THR = float(os.getenv("CONF_THR_ML_SENTIMENT",
                  os.getenv("CONF_THR", "0.80")))
R_MULT   = float(os.getenv("R_MULT_ML_SENTIMENT",
                  os.getenv("R_MULT", "2.0")))
SL_PCT   = float(os.getenv("SL_PCT_ML_SENTIMENT",
                  os.getenv("SL_PCT", "0.01")))          # 1% stop
SHORTS_ENABLED = os.getenv("SHORTS_ENABLED", "1").lower() in ("1","true","yes")

# Risk sizing (matches your previous semantics)
EQUITY_USD  = float(os.getenv("EQUITY_USD",  "100000"))
RISK_PCT    = float(os.getenv("RISK_PCT",    "0.01"))    # 1% risk budget
MAX_POS_PCT = float(os.getenv("MAX_POS_PCT", "0.05"))    # 5% notional cap
MIN_QTY     = int(os.getenv("MIN_QTY", "1"))
ROUND_LOT   = int(os.getenv("ROUND_LOT","1"))

def _risk_qty(entry_price: float, stop_price: float) -> int:
    if entry_price is None or stop_price is None:
        return 0
    risk_per_share = abs(float(entry_price) - float(stop_price))
    if risk_per_share <= 0:
        return 0
    qty_risk     = (EQUITY_USD * RISK_PCT) / risk_per_share
    qty_notional = (EQUITY_USD * MAX_POS_PCT) / max(1e-9, float(entry_price))
    qty = int(max(0, min(qty_risk, qty_notional)))
    if ROUND_LOT > 1:
        qty = (qty // ROUND_LOT) * ROUND_LOT
    return qty if qty > 0 else MIN_QTY

def _build(side: str, last: float, proba_up: float) -> Dict[str, Any]:
    # percent TP/SL (broker converts to absolute + snap/cushion)
    tp_pct = SL_PCT * R_MULT
    sl_pct = SL_PCT
    if side == "buy":
        sl_abs = last * (1 - sl_pct)
    else:  # sell/short
        sl_abs = last * (1 + sl_pct)
    qty = _risk_qty(last, sl_abs)
    if qty <= 0:
        return {}
    return {
        "side": side,
        "quantity": int(qty),
        "takeProfit": float(tp_pct),   # percent
        "stopLoss": float(sl_pct),     # percent
        "confidence": float(proba_up if side == "buy" else (1.0 - proba_up)),
    }

def decide(symbol: str, df1m: pd.DataFrame, tf_min: int) -> Optional[Dict[str, Any]]:
    # 1) require a model (prevents trades w/o load)
    model = maybe_load_model(STRAT)
    if model is None:
        return None

    # 2) features -> proba
    bars = resample_tf(df1m, tf_min)
    x_live, last = latest_features(bars)
    if x_live is None or last is None:
        return None

    p_up = proba_up_for(model, x_live)
    if p_up is None:
        return None

    want_long  = (p_up >= CONF_THR)
    want_short = ((1.0 - p_up) >= CONF_THR) and SHORTS_ENABLED

    if want_long:
        out = _build("buy", last, p_up)
        return out or None
    if want_short:
        out = _build("sell", last, p_up)
        return out or None

    return None
