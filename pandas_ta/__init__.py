# pandas_ta/__init__.py
# Minimal, local shim providing a subset of pandas_ta that your strategies likely use.
from __future__ import annotations
import numpy as np
import pandas as pd

__all__ = [
    "rsi", "ema", "sma", "macd", "bbands", "atr", "stoch", "vwap"
]

def _series(x, name=None):
    if isinstance(x, pd.Series):
        s = x.copy()
    else:
        s = pd.Series(x)
    if name:
        s.name = name
    return s

def sma(close: pd.Series, length: int = 20) -> pd.Series:
    close = _series(close)
    out = close.rolling(length, min_periods=length).mean()
    out.name = f"SMA_{length}"
    return out

def ema(close: pd.Series, length: int = 20) -> pd.Series:
    close = _series(close)
    out = close.ewm(span=length, adjust=False, min_periods=length).mean()
    out.name = f"EMA_{length}"
    return out

def rsi(close: pd.Series, length: int = 14) -> pd.Series:
    close = _series(close)
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1/length, adjust=False, min_periods=length).mean()
    avg_loss = loss.ewm(alpha=1/length, adjust=False, min_periods=length).mean()
    rs = avg_gain / (avg_loss + 1e-12)
    rsi = 100 - 100 / (1 + rs)
    rsi.name = f"RSI_{length}"
    return rsi

def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    close = _series(close)
    ema_fast = close.ewm(span=fast, adjust=False, min_periods=fast).mean()
    ema_slow = close.ewm(span=slow, adjust=False, min_periods=slow).mean()
    macd_line = ema_fast - ema_slow
    macd_signal = macd_line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    macd_hist = macd_line - macd_signal
    df = pd.DataFrame({
        f"MACD_{fast}_{slow}_{signal}": macd_line,
        f"MACDs_{fast}_{slow}_{signal}": macd_signal,
        f"MACDh_{fast}_{slow}_{signal}": macd_hist,
    })
    return df

def bbands(close: pd.Series, length: int = 20, std: float = 2.0) -> pd.DataFrame:
    close = _series(close)
    mid = sma(close, length)
    dev = close.rolling(length, min_periods=length).std(ddof=0)
    upper = mid + std * dev
    lower = mid - std * dev
    return pd.DataFrame({
        f"BBL_{length}_{int(std)}": lower,
        f"BBM_{length}_{int(std)}": mid,
        f"BBU_{length}_{int(std)}": upper,
    })

def atr(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 14) -> pd.Series:
    high = _series(high); low = _series(low); close = _series(close)
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low).abs(),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    out = tr.ewm(span=length, adjust=False, min_periods=length).mean()
    out.name = f"ATR_{length}"
    return out

def stoch(high: pd.Series, low: pd.Series, close: pd.Series, k: int = 14, d: int = 3, smooth_k: int = 3) -> pd.DataFrame:
    high = _series(high); low = _series(low); close = _series(close)
    ll = low.rolling(k, min_periods=k).min()
    hh = high.rolling(k, min_periods=k).max()
    k_raw = 100 * (close - ll) / (hh - ll + 1e-12)
    k_smooth = k_raw.rolling(smooth_k, min_periods=smooth_k).mean()
    d_line = k_smooth.rolling(d, min_periods=d).mean()
    return pd.DataFrame({
        f"STOCHk_{k}_{d}_{smooth_k}": k_smooth,
        f"STOCHd_{k}_{d}_{smooth_k}": d_line
    })

def vwap(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> pd.Series:
    high = _series(high); low = _series(low); close = _series(close); volume = _series(volume)
    typical = (high + low + close) / 3.0
    cum_pv = (typical * volume).cumsum()
    cum_v = (volume.replace(0, np.nan)).cumsum()
    out = cum_pv / cum_v
    out.name = "VWAP"
    return out
