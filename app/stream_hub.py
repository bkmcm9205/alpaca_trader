from __future__ import annotations
import asyncio, logging, threading
from typing import Callable, List
from .interfaces import Bar
from .config import Config
from .data_alpaca import run_stream

log = logging.getLogger("stream_hub")

class StreamHub:
    """
    Central stream -> fan-out callbacks.
    """
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.callbacks: List[Callable[[Bar], None]] = []
        self._thread: threading.Thread | None = None

    def register(self, cb: Callable[[Bar], None]):
        self.callbacks.append(cb)

    def _on_bar(self, bar: Bar):
        for cb in self.callbacks:
            try:
                cb(bar)
            except Exception:
                log.exception("[HUB] callback error")

    def start(self, symbols: List[str]):
        async def worker():
            await run_stream(self.cfg, symbols, self._on_bar)
        def runner():
            asyncio.run(worker())
        self._thread = threading.Thread(target=runner, daemon=True)
        self._thread.start()
        log.info(f"[HUB] started with {len(symbols)} symbols")

    def join(self):
        if self._thread:
            self._thread.join()
