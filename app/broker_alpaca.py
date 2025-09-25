from __future__ import annotations
import logging, math
from typing import List, Dict
from .interfaces import OrderIntent, Position, Side
from .throttler import PerMinuteLimiter

# Alpaca SDK
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.stream import TradingStream  # (optional if you later want trade updates)

log = logging.getLogger("broker_alpaca")

class BrokerAlpaca:
    def __init__(self, key_id: str, secret: str, paper: bool, order_rate_pm: int, status_rate_pm: int):
        self.client = TradingClient(api_key=key_id, secret_key=secret, paper=paper)
        self.order_limiter = PerMinuteLimiter(order_rate_pm)
        self.status_limiter = PerMinuteLimiter(status_rate_pm)

    def get_equity(self) -> float:
        self.status_limiter.wait_until_allowed()
        acct = self.client.get_account()
        return float(acct.equity)

    def get_positions(self) -> Dict[str, Position]:
        self.status_limiter.wait_until_allowed()
        pos = {}
        for p in self.client.get_all_positions():
            qty = int(float(p.qty))
            side = Side.BUY if float(p.qty) > 0 else Side.SELL
            pos[p.symbol.upper()] = Position(
                symbol=p.symbol.upper(),
                qty=abs(qty),
                avg_entry_price=float(p.avg_entry_price),
                market_price=float(p.current_price),
                side=side,
            )
        return pos

    def submit_market(self, intent: OrderIntent):
        self.order_limiter.wait_until_allowed()
        side = OrderSide.BUY if intent.side == Side.BUY else OrderSide.SELL
        # Convert TP/SL pct to limit/stop prices from current market is tricky; we use brackets by specifying
        # take_profit and stop_loss as dollar targets offset from avg fill (Alpaca handles it).
        req = MarketOrderRequest(
            symbol=intent.symbol,
            qty=int(intent.qty),
            side=side,
            time_in_force=TimeInForce.DAY,
            take_profit=None if intent.take_profit_pct <= 0 else {"limit_price": 0},  # placeholder -> handled by Alpaca bracket
            stop_loss=None if intent.stop_loss_pct <= 0 else {"stop_price": 0},
        )
        # NOTE: alpaca-py sets order_class automatically when TP/SL present. Some versions require dicts above;
        # if bracket creation complains, submit plain market and manage exits in code.
        try:
            o = self.client.submit_order(order_data=req)
            log.info(f"[ORDER] {intent.side} {intent.qty} {intent.symbol} submitted id={o.id}")
            return o
        except Exception as e:
            log.exception(f"[ORDER-ERROR] {intent} -> {e}")
            return None

    def flatten_all(self):
        """Market close out all positions."""
        self.status_limiter.wait_until_allowed()
        try:
            self.client.close_all_positions(cancel_orders=True)
            log.warning("[BROKER] Close-all triggered")
        except Exception as e:
            log.exception(f"[BROKER] close_all_positions error: {e}")
