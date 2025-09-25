from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Any
from datetime import datetime
from enum import Enum


class Side(str, Enum):
    BUY = "buy"
    SELL = "sell"


@dataclass
class Bar:
    symbol: str
    ts: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass
class Signal:
    symbol: str
    side: Optional[Side]  # None = no action/flat
    confidence: float = 0.0
    take_profit_pct: float = 0.0  # e.g., 0.03 for +3%
    stop_loss_pct: float = 0.0    # e.g., 0.01 for -1%
    metadata: Dict[str, Any] = None


@dataclass
class OrderIntent:
    symbol: str
    qty: int
    side: Side
    time_in_force: str = "day"
    take_profit_pct: float = 0.0
    stop_loss_pct: float = 0.0
    meta: Dict[str, Any] = None


@dataclass
class Position:
    symbol: str
    qty: int
    avg_entry_price: float
    market_price: float
    side: Side  # BUY(long) or SELL(short)
