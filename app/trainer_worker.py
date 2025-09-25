from __future__ import annotations
import time, logging
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from alpaca.data.timeframe import TimeFrame

from .config import Config
from .logutil import setup_logging
from .model_registry_s3 import S3ModelRegistry
from .data_alpaca import fetch_historical_bars

log = logging.getLogger("trainer")

def train_dummy_momentum(bars: pd.DataFrame):
    """
    Make a trivial 'model': decide BUY if 1-min return > 0 (used only as example).
    We'll just store thresholds in a dict to pickle.
    """
    ret = bars["close"].pct_change().fillna(0.0)
    metric = float((ret > 0).mean())  # 'accuracy' vs naive
    model = {"type":"dummy_momo", "conf_scale": float(ret.abs().median())}
    metrics = {"val_sharpe": metric, "acc": metric}
    return model, metrics

class DummyModelWrapper:
    def __init__(self, conf_scale: float):
        self.conf_scale = conf_scale

    def predict(self, bar):
        from .interfaces import Signal, Side
        conf = abs(bar.close - bar.open) / max(1e-3, self.conf_scale)
        side = Side.BUY if bar.close > bar.open else None
        return Signal(symbol=bar.symbol, side=side, confidence=float(conf), take_profit_pct=0.03, stop_loss_pct=0.01)

def main():
    cfg = Config.load_from_env()
    setup_logging(cfg.diag_verbose)
    reg = S3ModelRegistry(cfg.s3_bucket, cfg.s3_region, cfg.s3_base_prefix)
    tz = ZoneInfo("America/New_York")

    log.info("[BOOT] Trainer worker up (daily loop)")

    while True:
        now = datetime.now(tz)
        # Train daily at 02:05 ET
        if now.hour == 2 and now.minute >= 5 and now.minute < 20:
            try:
                symbols = cfg.hub_symbols[:50]  # small demo universe
                end = datetime.now(timezone.utc)
                start = end - timedelta(days=5)
                data = fetch_historical_bars(cfg, symbols, start, end, timeframe=TimeFrame.Minute)

                # concat into one DF
                frames = []
                for sym, bars in data.items():
                    for b in bars:
                        frames.append([sym, b.ts, b.open, b.high, b.low, b.close, b.volume])
                if not frames:
                    time.sleep(30); continue
                df = pd.DataFrame(frames, columns=["symbol","ts","open","high","low","close","volume"])
                df = df.sort_values(["symbol","ts"]).set_index("ts")

                model, metrics = train_dummy_momentum(df)
                # Wrap model so strategy_runner can call .predict(bar)
                model_obj = DummyModelWrapper(conf_scale=float(model["conf_scale"]))

                for name in cfg.strategies.keys():
                    prefix = reg.save_candidate(name, model_obj, metrics)
                    promoted, cand_val, prod_val = reg.maybe_promote_if_better(name, prefix, "val_sharpe")
                    log.info(f"[TRAIN] {name} {('PROMOTED' if promoted else 'kept')} candidate (cand={cand_val:.4f} vs prod={prod_val:.4f})")

            except Exception:
                log.exception("[TRAIN] failure during training cycle")

            # sleep out the window
            time.sleep(60*30)
        else:
            time.sleep(20)

if __name__ == "__main__":
    main()
