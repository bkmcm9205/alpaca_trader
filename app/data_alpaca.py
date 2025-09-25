from __future__ import annotations
import asyncio, logging, sys
from datetime import datetime, timedelta, timezone
from typing import Callable, List, Dict

from .interfaces import Bar
from .config import Config
from .logutil import setup_logging

# Alpaca data SDK (historical + live)
try:
    from alpaca.data.historical import StockHistoricalDataClient
except Exception:
    # older/newer layout:
    from alpaca.data import StockHistoricalDataClient  # type: ignore

from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.live import StockDataStream

log = logging.getLogger("data_alpaca")

def bars_df_to_bar_objects(symbol: str, df) -> List[Bar]:
    rows = []
    for idx, r in df.iterrows():
        rows.append(Bar(
            symbol=symbol,
            ts=idx.to_pydatetime() if hasattr(idx, "to_pydatetime") else idx,
            open=float(r["open"]),
            high=float(r["high"]),
            low=float(r["low"]),
            close=float(r["close"]),
            volume=float(r["volume"]),
        ))
    return rows

def fetch_historical_bars(cfg: Config, symbols: List[str], start: datetime, end: datetime, timeframe: TimeFrame=TimeFrame.Minute) -> Dict[str, List[Bar]]:
    client = StockHistoricalDataClient(api_key=cfg.alpaca_key_id, secret_key=cfg.alpaca_secret_key)
    req = StockBarsRequest(symbols=symbols, timeframe=timeframe, start=start, end=end, feed=cfg.data_feed)
    out: Dict[str, List[Bar]] = {}
    data = client.get_stock_bars(req)
    # alpaca-py returns a dataframe via .df in many versions
    try:
        df = data.df
        for sym in symbols:
            sdf = df.loc[df.index.get_level_values("symbol") == sym]
            out[sym] = bars_df_to_bar_objects(sym, sdf.droplevel("symbol"))
    except Exception:
        # fallback: iterate dict-of-dfs
        for sym, sdf in data.items():
            out[sym] = bars_df_to_bar_objects(sym, sdf)
    return out

async def run_stream(cfg: Config, symbols: List[str], on_bar: Callable[[Bar], None]):
    stream = StockDataStream(cfg.alpaca_key_id, cfg.alpaca_secret_key, feed=cfg.data_feed)
    async def _handler(b):
        try:
            bar = Bar(
                symbol=b.symbol.upper(),
                ts=b.timestamp if hasattr(b, "timestamp") else datetime.now(timezone.utc),
                open=float(b.open),
                high=float(b.high),
                low=float(b.low),
                close=float(b.close),
                volume=float(b.volume),
            )
            on_bar(bar)
        except Exception as e:
            log.exception(f"[STREAM] handler error: {e}")

    for sym in symbols:
        stream.subscribe_bars(_handler, sym)

    log.info(f"[STREAM] subscribing minute bars for {len(symbols)} symbols via {cfg.data_feed.upper()}")
    await stream.run()

# -------- CLI: simple backfill --------
if __name__ == "__main__":
    import argparse
    setup_logging(True)
    cfg = Config.load_from_env()

    p = argparse.ArgumentParser()
    p.add_argument("--symbols", type=str, required=True, help="CSV symbols e.g. AAPL,MSFT,SPY")
    p.add_argument("--days", type=int, default=5)
    args = p.parse_args()

    syms = [s.strip().upper() for s in args.symbols.split(",")]
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=args.days)
    data = fetch_historical_bars(cfg, syms, start, end)
    total = sum(len(v) for v in data.values())
    log.info(f"Fetched {total} bars across {len(data)} symbols from {start} -> {end}")
