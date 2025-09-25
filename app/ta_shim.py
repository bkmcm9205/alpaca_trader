# app/ta_shim.py
from __future__ import annotations
import numpy as np
import pandas as pd

__all__ = ["rsi","ema","sma","macd","bbands","atr","stoch","vwap"]

def _S(x, name=None):
    s = x.copy() if isinstance(x, pd.Series) else pd.Series(x)
    if name: s.name = name
    return s

def sma(close: pd.Series, length: int = 20) -> pd.Series:
    close = _S(close); out = close.rolling(length, min_periods=length).mean()
    out.name = f"SMA_{length}"; return out

def ema(close: pd.Series, length: int = 20) -> pd.Series:
    close = _S(close); out = close.ewm(span=length, adjust=False, min_periods=length).mean()
    out.name = f"EMA_{length}"; return out

def rsi(close: pd.Series, length: int = 14) -> pd.Series:
    close=_S(close); d=close.diff(); g=d.clip(lower=0); l=(-d).clip(lower=0)
    ag=g.ewm(alpha=1/length, adjust=False, min_periods=length).mean()
    al=l.ewm(alpha=1/length, adjust=False, min_periods=length).mean()
    rs=ag/(al+1e-12); out=100-100/(1+rs); out.name=f"RSI_{length}"; return out

def macd(close: pd.Series, fast:int=12, slow:int=26, signal:int=9) -> pd.DataFrame:
    close=_S(close); ef=close.ewm(span=fast, adjust=False, min_periods=fast).mean()
    es=close.ewm(span=slow, adjust=False, min_periods=slow).mean()
    line=ef-es; sig=line.ewm(span=signal, adjust=False, min_periods=signal).mean(); hist=line-sig
    return pd.DataFrame({f"MACD_{fast}_{slow}_{signal}":line, f"MACDs_{fast}_{slow}_{signal}":sig, f"MACDh_{fast}_{slow}_{signal}":hist})

def bbands(close: pd.Series, length:int=20, std:float=2.0) -> pd.DataFrame:
    close=_S(close); mid=sma(close,length); dev=close.rolling(length, min_periods=length).std(ddof=0)
    upper=mid+std*dev; lower=mid-std*dev
    return pd.DataFrame({f"BBL_{length}_{int(std)}":lower, f"BBM_{length}_{int(std)}":mid, f"BBU_{length}_{int(std)}":upper})

def atr(high: pd.Series, low: pd.Series, close: pd.Series, length:int=14) -> pd.Series:
    high=_S(high); low=_S(low); close=_S(close); pc=close.shift(1)
    tr=pd.concat([(high-low).abs(), (high-pc).abs(), (low-pc).abs()], axis=1).max(axis=1)
    out=tr.ewm(span=length, adjust=False, min_periods=length).mean(); out.name=f"ATR_{length}"; return out

def stoch(high: pd.Series, low: pd.Series, close: pd.Series, k:int=14, d:int=3, smooth_k:int=3) -> pd.DataFrame:
    high=_S(high); low=_S(low); close=_S(close)
    ll=low.rolling(k, min_periods=k).min(); hh=high.rolling(k, min_periods=k).max()
    kraw=100*(close-ll)/(hh-ll+1e-12); ksm=kraw.rolling(smooth_k, min_periods=smooth_k).mean()
    dline=ksm.rolling(d, min_periods=d).mean()
    return pd.DataFrame({f"STOCHk_{k}_{d}_{smooth_k}":ksm, f"STOCHd_{k}_{d}_{smooth_k}":dline})

def vwap(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> pd.Series:
    high=_S(high); low=_S(low); close=_S(close); volume=_S(volume)
    typical=(high+low+close)/3.0; cpv=(typical*volume).cumsum(); cv=(volume.replace(0,np.nan)).cumsum()
    out=cpv/cv; out.name="VWAP"; return out
