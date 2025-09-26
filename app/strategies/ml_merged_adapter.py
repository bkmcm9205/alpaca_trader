# app/strategies/ml_merged_adapter.py
from __future__ import annotations
import os
from typing import Optional, Dict, Any

import pandas as pd

from app.strategies.model_runtime import (
    maybe_load_model, resample_tf, latest_features, proba_up_for
)

STRAT = "ml_merged"

CONF_THR = float(os.getenv("CONF_THR_ML_MERGED", os.getenv("CONF_THR", "0.80")))
R_MULT   = float(os.getenv("R_MULT_ML_MERGED",   os.getenv("R_MULT", "3.0")))
SHORTS_ENABLED = os.getenv("SHORTS_ENABLED", "1").lower() in ("1","true","yes")

def _build_payload(side: str, last: float, proba_up: float) -> Dict[str, Any]:
    sl_pct = 0.01
    tp_pct = 0.01 * R_MULT
    return {
        "side": side,
        "quantity": None,
        "takeProfit": tp_pct,
        "stopLoss": sl_pct,
        "confidence": float(proba_up if side == "buy" else (1.0 - proba_up)),
    }

def decide(symbol: str, df1m: pd.DataFrame, tf_min: int) -> Optional[Dict[str, Any]]:
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
