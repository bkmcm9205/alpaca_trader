from __future__ import annotations
import importlib, json
from typing import List, Callable, Any, Dict
from ..interfaces import Signal, Side, Bar

class FunctionAdapter:
    """
    Wraps an arbitrary function defined at module:function.
    The function should accept: (bar_dict, close_history_list) and return:
      {
        "side": "buy" | "sell" | None,
        "confidence": float,
        "tp_pct": float | 0,
        "sl_pct": float | 0
      }
    """
    def __init__(self, callable: str, **kwargs):
        mod_path, func_name = callable.split(":")
        mod = importlib.import_module(mod_path)
        self.fn: Callable[[Dict[str, Any], List[float]], Dict[str, Any]] = getattr(mod, func_name)
        self.extra = kwargs  # ignored by default, there if you need params

    def predict(self, bar: Bar, close_history: List[float]) -> Signal:
        b = {
            "symbol": bar.symbol,
            "ts": bar.ts.isoformat() if hasattr(bar.ts, "isoformat") else bar.ts,
            "open": float(bar.open),
            "high": float(bar.high),
            "low": float(bar.low),
            "close": float(bar.close),
            "volume": float(bar.volume),
        }
        out = self.fn(b, list(close_history)) or {}
        side = out.get("side")
        if side is None:
            return Signal(symbol=bar.symbol, side=None, confidence=0.0)
        side_enum = Side.BUY if str(side).lower() == "buy" else Side.SELL
        return Signal(
            symbol=bar.symbol,
            side=side_enum,
            confidence=float(out.get("confidence", 0.0)),
            take_profit_pct=float(out.get("tp_pct", 0.0)),
            stop_loss_pct=float(out.get("sl_pct", 0.0)),
        )
