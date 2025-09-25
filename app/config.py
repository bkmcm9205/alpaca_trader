from __future__ import annotations
import os, json
from dataclasses import dataclass, field
from typing import List, Dict
from datetime import time


def _as_bool(v: str, default: bool=False) -> bool:
    if v is None: return default
    return str(v).strip().lower() in ("1","true","yes","y","on")

def _as_int(v: str, default: int) -> int:
    try:
        return int(str(v).strip())
    except Exception:
        return default

def _split_csv(v: str) -> List[str]:
    return [s.strip().upper() for s in str(v or "").replace(";",",").split(",") if s.strip()]

@dataclass
class StrategyCfg:
    name: str
    symbols: List[str] = field(default_factory=list)   # if empty, use HUB default
    model_key: str = ""                                # s3 key for production model (prefix only)
    conf_thr: float = 0.7
    r_mult: float = 3.0
    max_positions: int = 200
    min_price: float = 3.0
    min_today_vol: int = 20_000
    allow_shorts: bool = False

@dataclass
class Config:
    # Alpaca
    alpaca_key_id: str
    alpaca_secret_key: str
    alpaca_paper: bool = True
    data_feed: str = "sip"  # "iex" or "sip"
    # S3
    s3_bucket: str = ""
    s3_region: str = ""
    s3_base_prefix: str = "models"
    # Universe / streaming
    hub_symbols: List[str] = field(default_factory=lambda: ["SPY","QQQ","AAPL","MSFT","NVDA"])
    subscribe_timeframe: str = "1Min"  # visual only; streaming is minute bars
    # Guards
    start_equity_usd: float = 100000.0
    pt_up_pct: float = 0.10     # +10% halt day (profit target)
    dd_down_pct: float = 0.05   # -5% halt day (drawdown)
    kill_on_guard: bool = True  # if True: flatten + stop entries for the day
    # EOD
    eod_flatten_start_et: str = "16:00"
    eod_flatten_end_et: str = "16:10"
    # Throttles
    order_rate_per_min: int = 120
    status_rate_per_min: int = 240
    # Strategies
    strategies: Dict[str, StrategyCfg] = field(default_factory=dict)
    # Logging
    diag_verbose: bool = False

    @staticmethod
    def load_from_env() -> "Config":
        # Core
        key = os.getenv("ALPACA_KEY_ID","")
        secret = os.getenv("ALPACA_SECRET_KEY","")
        paper = _as_bool(os.getenv("ALPACA_PAPER","true"), True)
        feed = os.getenv("ALPACA_DATA_FEED","sip").lower()

        s3_bucket = os.getenv("S3_BUCKET","")
        s3_region = os.getenv("S3_REGION","")
        s3_base = os.getenv("S3_BASE_PREFIX","models")

        hub_symbols = _split_csv(os.getenv("SYMBOLS", "SPY,QQQ,AAPL,MSFT,NVDA"))

        start_eq = float(os.getenv("START_EQUITY", os.getenv("EQUITY_USD", "100000")))
        pt = float(os.getenv("DAILY_PROFIT_TARGET_PCT", "0.10"))
        dd = float(os.getenv("DAILY_DRAWDOWN_PCT", "0.05"))
        kill = _as_bool(os.getenv("GUARD_FLATTEN", "true"), True)

        eod_s = os.getenv("EOD_FLATTEN_START_ET", "16:00")
        eod_e = os.getenv("EOD_FLATTEN_END_ET", "16:10")

        orl = _as_int(os.getenv("ORDER_RATE_LIMIT_PM","120"), 120)
        srl = _as_int(os.getenv("STATUS_RATE_LIMIT_PM","240"), 240)

        diag = _as_bool(os.getenv("DIAG_VERBOSE","false"), False)

        # Strategies (env-only, no code edits)
        names = _split_csv(os.getenv("STRATEGY_NAMES","ranked_ml"))
        strategies = {}
        for name in names:
            prefix = f"S_{name.upper()}_"
            strategies[name] = StrategyCfg(
                name=name,
                symbols=_split_csv(os.getenv(prefix+"SYMBOLS","")),
                model_key=os.getenv(prefix+"MODEL_KEY",""),
                conf_thr=float(os.getenv(prefix+"CONF_THR","0.7")),
                r_mult=float(os.getenv(prefix+"R_MULT","3.0")),
                max_positions=_as_int(os.getenv(prefix+"MAX_POSITIONS","200"), 200),
                min_price=float(os.getenv(prefix+"MIN_PRICE","3.0")),
                min_today_vol=_as_int(os.getenv(prefix+"MIN_TODAY_VOL","20000"), 20000),
                allow_shorts=_as_bool(os.getenv(prefix+"SHORTS_ENABLED","false"), False),
            )

        return Config(
            alpaca_key_id=key,
            alpaca_secret_key=secret,
            alpaca_paper=paper,
            data_feed=feed,
            s3_bucket=s3_bucket,
            s3_region=s3_region,
            s3_base_prefix=s3_base,
            hub_symbols=hub_symbols,
            start_equity_usd=start_eq,
            pt_up_pct=pt,
            dd_down_pct=dd,
            kill_on_guard=kill,
            eod_flatten_start_et=eod_s,
            eod_flatten_end_et=eod_e,
            order_rate_per_min=orl,
            status_rate_per_min=srl,
            strategies=strategies,
            diag_verbose=diag,
        )
