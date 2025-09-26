# app/strategies/trading_bot_adapter.py
from __future__ import annotations
import os
from typing import Optional, Dict, Any

import pandas as pd

from app.strategies.model_runtime import (
    maybe_load_model, resample_tf, latest_features, proba_up_for
)

STRAT = "trading_bot"

# Env knobs (per your runner/infra)
CONF_THR = float(os.getenv("CONF_THR_TRADING_BOT", os.getenv("CONF_THR", "0.80")))
R_MULT   = float(os.getenv("R_MULT_TRADING_BOT",   os.getenv("R_MULT", "3.0")))
SHORTS_ENABLED = os.getenv("SHORTS_ENABLED", "1").lower() in ("1","true","yes")

def _build_payload(side: str, last: float, proba_up: float) -> Dict[str, Any]:
    """
    Return a runner-compatible signal dict.
    - Quantities: let runner fill with MIN_QTY if not provided.
    - TP/SL as percentages (broker will translate around last).
    """
    # 1% risk unit; TP = r_mult * 1%
    sl_pct = 0.01
    tp_pct = 0.01 * R_MULT
    return {
        "side": side,                 # "buy" or "sell"
        "quantity": None,             # let runner normalize to MIN_QTY
        "takeProfit": tp_pct,         # percentage
        "stopLoss": sl_pct,           # percentage
        "confidence": float(proba_up if side == "buy" else (1.0 - proba_up)),
    }

def decide(symbol: str, df1m: pd.DataFrame, tf_min: int) -> Optional[Dict[str, Any]]:
    """
    Called by strategy_runner. Return a signal dict or None.
    """
    # Load model if we’re in pre-market reload window
    model = maybe_load_model(STRAT)

    bars = resample_tf(df1m, tf_min)
    x_live, last = latest_features(bars)
    if x_live is None or last is None:
        return None

    p_up = proba_up_for(model, x_live)
    if p_up is None:
        return None

    want_long  = p_up >= CONF_THR
    want_short = (1.0 - p_up) >= CONF_THR and SHORTS_ENABLED

    if want_long:
        return _build_payload("buy", last, p_up)
    if want_short:
        return _build_payload("sell", last, p_up)

    return None
