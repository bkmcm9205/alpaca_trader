from __future__ import annotations
import os, time, logging
from typing import Dict
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import numpy as np

from .config import Config, StrategyCfg
from .logutil import setup_logging
from .interfaces import Bar, Side, Signal, OrderIntent
from .stream_hub import StreamHub
from .broker_alpaca import BrokerAlpaca
from .model_registry_s3 import S3ModelRegistry
from .throttler import PerMinuteLimiter

log = logging.getLogger("strategy_runner")

# ----- Simple fallback "model" if none in S3 -----
class SimpleMomentumModel:
    def predict(self, bar: Bar) -> Signal:
        # toy: if close made a new minute high vs open -> BUY; else no action
        side = Side.BUY if bar.close > bar.open else None
        conf = abs(bar.close - bar.open) / max(bar.open, 1e-6)
        return Signal(symbol=bar.symbol, side=side, confidence=float(conf), take_profit_pct=0.03, stop_loss_pct=0.01)

# ----- Per-strategy engine -----
class Engine:
    def __init__(self, cfg: Config, scfg: StrategyCfg, broker: BrokerAlpaca, registry: S3ModelRegistry):
        self.cfg = cfg
        self.scfg = scfg
        self.broker = broker
        self.registry = registry
        self.model = self._load_model()
        self.et = ZoneInfo("America/New_York")
        self.open_positions: Dict[str,int] = {}
        self.no_more_entries_today = False
        self.order_throttle = PerMinuteLimiter(max(1, cfg.order_rate_per_min // max(1, len(cfg.strategies))))

        log.info(f"[{scfg.name}] engine ready. shorts={scfg.allow_shorts} conf_thr={scfg.conf_thr} max_pos={scfg.max_positions}")

    def _load_model(self):
        try:
            if self.scfg.model_key:
                # allow explicit key override like models/<strategy>/production/model.pkl
                s3key = self.scfg.model_key
                # If it's a prefix, use registry loader; else direct get_object is also fine
            return self.registry.load_production(self.scfg.name)
        except Exception as e:
            log.warning(f"[{self.scfg.name}] no production model, using SimpleMomentumModel. ({e})")
            return SimpleMomentumModel()

    def _guard_window_now(self):
        now_et = datetime.now(self.et).time()
        s = self.cfg.eod_flatten_start_et
        e = self.cfg.eod_flatten_end_et
        return s <= now_et.strftime("%H:%M") <= e

    def _daily_guard_check(self) -> bool:
        """True if guard breached; handles flatten+halt if configured."""
        try:
            eq = self.broker.get_equity()
        except Exception:
            log.exception("[GUARD] could not fetch equity; skipping")
            return False
        up_target = self.cfg.start_equity_usd * (1.0 + self.cfg.pt_up_pct)
        dn_target = self.cfg.start_equity_usd * (1.0 - self.cfg.dd_down_pct)
        if eq >= up_target:
            log.warning(f"[GUARD] Profit target met: eq={eq:.2f} >= {up_target:.2f}")
            if self.cfg.kill_on_guard:
                self.broker.flatten_all()
                self.no_more_entries_today = True
            return True
        if eq <= dn_target:
            log.warning(f"[GUARD] Drawdown breached: eq={eq:.2f} <= {dn_target:.2f}")
            if self.cfg.kill_on_guard:
                self.broker.flatten_all()
                self.no_more_entries_today = True
            return True
        return False

    def process_bar(self, bar: Bar):
        try:
            # EOD flatten window
            if self._guard_window_now():
                self.broker.flatten_all()
                self.no_more_entries_today = True
                return

            # Daily guard
            if self._daily_guard_check():
                return

            # If halted, do not enter
            if self.no_more_entries_today:
                return

            # Basic symbol gates
            if bar.close < self.scfg.min_price:
                return

            # Model decision
            sig: Signal = self.model.predict(bar)
            if sig.side is None or sig.confidence < self.scfg.conf_thr:
                return

            # Position gating
            if len(self.open_positions) >= self.scfg.max_positions:
                return

            # Position sizing: naive fixed notional / price
            notional = max(1000.0, self.cfg.start_equity_usd * 0.005)  # ~0.5% slice
            qty = int(max(1, notional // max(bar.close, 1.0)))

            # Throttle orders
            if not self.order_throttle.allow():
                return

            oi = OrderIntent(
                symbol=bar.symbol,
                qty=qty,
                side=sig.side,
                take_profit_pct=float(sig.take_profit_pct),
                stop_loss_pct=float(sig.stop_loss_pct),
            )
            o = self.broker.submit_market(oi)
            if o:
                self.open_positions[bar.symbol] = self.open_positions.get(bar.symbol, 0) + qty

        except Exception:
            log.exception(f"[{self.scfg.name}] process_bar error")

def main():
    cfg = Config.load_from_env()
    setup_logging(cfg.diag_verbose)

    log.info("[BOOT] Strategy Runner starting…")
    broker = BrokerAlpaca(cfg.alpaca_key_id, cfg.alpaca_secret_key, cfg.alpaca_paper,
                          cfg.order_rate_per_min, cfg.status_rate_per_min)
    registry = S3ModelRegistry(cfg.s3_bucket, cfg.s3_region, cfg.s3_base_prefix)

    hub = StreamHub(cfg)

    # Instantiate one Engine per strategy
    engines = []
    for name, scfg in cfg.strategies.items():
        eng = Engine(cfg, scfg, broker, registry)
        engines.append(eng)
        hub.register(eng.process_bar)

    # Symbols: union of per-strategy overrides or hub default
    all_syms = set(cfg.hub_symbols)
    for scfg in cfg.strategies.values():
        if scfg.symbols:
            all_syms.update(scfg.symbols)

    hub.start(sorted(all_syms))
    hub.join()

if __name__ == "__main__":
    main()
