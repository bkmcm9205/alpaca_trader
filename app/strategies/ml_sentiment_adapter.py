# app/strategies/ml_sentiment_adapter.py
# Adapter for ml_sentiment: sentiment-gated ML pattern with per-share-risk sizing
# - Keeps your CONF_THR / R_MULT / SHORTS_ENABLED logic
# - Uses Alpaca data via data_alpaca.fetch_alpaca_1m()
# - Emits order intents (symbol, side, qty, tp, sl, reason, meta)
# - No TradersPost/Polygon; guard rails & EOD flatten handled by strategy_runner

from __future__ import annotations
from typing import Iterable, Dict, Any, List, Optional
import os, math, hashlib
from datetime import datetime, timezone
import numpy as np

try:
    import pandas as pd
except Exception:
    pd = None

from app.data_alpaca import fetch_alpaca_1m  # must return ET-indexed 1m bars with o/h/l/c/volume
# TA shim is injected at launch if pandas_ta is missing; we attempt to use it if present
try:
    import pandas_ta as ta  # will be provided by ta_shim if not installed
except Exception:
    ta = None

# ============
# ENV / KNOBS
# ============
MARKET_TZ = os.getenv("MARKET_TZ", "America/New_York")

# Exchanges / filters
ALLOWED_EXCHANGES = set(
    x.strip().upper()
    for x in os.getenv("ALLOWED_EXCHANGES", "NASD,NASDAQ,NYSE,XNAS,XNYS").split(",")
    if x.strip()
)
MIN_PRICE = float(os.getenv("MIN_PRICE", "3.0"))

# Scanner volume + timeframes
SCANNER_MIN_TODAY_VOL = int(os.getenv("SCANNER_MIN_TODAY_VOL", "100000"))
TF_MIN_LIST = [int(x) for x in os.getenv("TF_MIN_LIST", "1,2,3,5,10").split(",") if x.strip()]
LOOKBACK_MIN = int(os.getenv("LOOKBACK_MIN", "2400"))
MAX_UNIVERSE_PAGES = int(os.getenv("MAX_UNIVERSE_PAGES", "7"))  # runner handles universe; here for parity

# Sizing (per-share risk, notional caps)
EQUITY_USD = float(os.getenv("EQUITY_USD", "100000"))
RISK_PCT   = float(os.getenv("RISK_PCT", "0.01"))    # 1%
MAX_POS_PCT= float(os.getenv("MAX_POS_PCT","0.10"))  # 10%
MIN_QTY    = int(os.getenv("MIN_QTY", "1"))
ROUND_LOT  = int(os.getenv("ROUND_LOT","1"))

# Sentiment regime
SENTIMENT_LOOKBACK_MIN = int(os.getenv("SENTIMENT_LOOKBACK_MIN", "5"))
SENTIMENT_NEUTRAL_BAND = float(os.getenv("SENTIMENT_NEUTRAL_BAND","0.0015"))
SENTIMENT_SYMBOLS = [s.strip() for s in os.getenv("SENTIMENT_SYMBOLS","SPY,QQQ").split(",") if s.strip()]
USE_SENTIMENT_REGIME = os.getenv("USE_SENTIMENT_REGIME","1").lower() in ("1","true","yes")

# Session windows
ALLOW_PREMARKET  = os.getenv("ALLOW_PREMARKET","0").lower() in ("1","true","yes")
ALLOW_AFTERHOURS = os.getenv("ALLOW_AFTERHOURS","0").lower() in ("1","true","yes")

# Model thresholds
# Use strategy-specific override first, then default, then base
def _env_float(*keys: str, default: float) -> float:
    for k in keys:
        v = os.getenv(k)
        if v is not None and str(v).strip() != "":
            try:
                return float(v)
            except Exception:
                pass
    return float(default)

CONF_THR = _env_float("CONF_THR_ML_SENTIMENT","CONF_THR_DEFAULT","CONF_THR", default=0.80)
R_MULT   = _env_float("R_MULT", default=3.0)
SHORTS_ENABLED = os.getenv("SHORTS_ENABLED","1").lower() in ("1","true","yes")

# Misc
DEDUP_BY_SYMBOL = os.getenv("DEDUP_BY_SYMBOL","1").lower() in ("1","true","yes")
POLL_SECONDS = int(os.getenv("POLL_SECONDS","10"))

# =========================
# Helpers: time & sessions
# =========================
def _now_local(ctx) -> datetime:
    # runner should expose ctx.now (aware); fall back to UTC now
    return (getattr(ctx, "now", None) or datetime.now(timezone.utc)).astimezone()

def _is_rth(ts: datetime) -> bool:
    # 9:30–16:00 local ET assumed
    h, m = ts.hour, ts.minute
    return ((h > 9) or (h == 9 and m >= 30)) and (h < 16)

def _in_session(ts: datetime) -> bool:
    if _is_rth(ts):
        return True
    if ALLOW_PREMARKET and (4 <= ts.hour < 9 or (ts.hour == 9 and ts.minute < 30)):
        return True
    if ALLOW_AFTERHOURS and (16 <= ts.hour < 20):
        return True
    return False

# =========================
# Sizing by per-share risk
# =========================
def _position_qty(entry_price: float, stop_price: float,
                  equity=EQUITY_USD, risk_pct=RISK_PCT, max_pos_pct=MAX_POS_PCT,
                  min_qty=MIN_QTY, round_lot=ROUND_LOT) -> int:
    if entry_price is None or stop_price is None:
        return 0
    risk_per_share = abs(float(entry_price) - float(stop_price))
    if risk_per_share <= 0:
        return 0
    qty_risk     = (equity * risk_pct) / risk_per_share
    qty_notional = (equity * max_pos_pct) / max(1e-9, float(entry_price))
    qty = math.floor(max(min(qty_risk, qty_notional), 0) / max(1, round_lot)) * max(1, round_lot)
    return int(max(qty, min_qty if qty > 0 else 0))

# =========================
# Feature model (quick RF)
# =========================
def _ml_features_and_pred(bars: pd.DataFrame) -> tuple[Optional[datetime], Optional[float], Optional[int]]:
    """
    Train quick RF on rolling features; return (timestamp, proba_up, pred_up) for last bar.
    Falls back to simple momentum if sklearn not available.
    """
    if bars is None or bars.empty or not isinstance(bars.index, pd.DatetimeIndex):
        return None, None, None

    # Require some history
    if len(bars) < 120:
        return None, None, None

    df = bars.copy()
    df["return"] = df["close"].pct_change()

    # RSI via pandas_ta if available; otherwise manual approximation
    if ta is not None and hasattr(ta, "rsi"):
        try:
            df["rsi"] = ta.rsi(df["close"], length=14)
        except Exception:
            df["rsi"] = df["close"].pct_change().rolling(14).apply(lambda x: 50 + 50*np.tanh(x.mean()*100), raw=True)
    else:
        delta = df["close"].diff()
        up = delta.clip(lower=0).rolling(14).mean()
        down = (-delta.clip(upper=0)).rolling(14).mean()
        rs = up / (down.replace(0, np.nan))
        df["rsi"] = 100 - (100 / (1 + rs))

    df["volatility"] = df["close"].rolling(20).std()
    df.dropna(inplace=True)
    if len(df) < 60:
        return None, None, None

    X = df[["return","rsi","volatility"]].iloc[:-1]
    y = (df["close"].shift(-1) > df["close"]).astype(int).iloc[:-1]
    if len(X) < 50:
        return None, None, None

    try:
        from sklearn.ensemble import RandomForestClassifier
        cut = int(len(X) * 0.7)
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X.iloc[:cut], y.iloc[:cut])
        x_live = X.iloc[[-1]]
        try:
            x_live = x_live[list(clf.feature_names_in_)]
        except Exception:
            pass
        proba_up = float(clf.predict_proba(x_live)[0][1])
    except Exception:
        # Fallback: probability proxy via momentum & RSI
        mom = float(df["close"].iloc[-1] / df["close"].iloc[-10] - 1.0) if len(df) >= 10 else 0.0
        rsi = float(df["rsi"].iloc[-1])
        proba_up = max(0.0, min(1.0, 0.5 + 0.4*mom + 0.1*((rsi-50)/50)))

    pred_up  = int(proba_up >= 0.5)
    ts = df.index[-1].to_pydatetime()
    return ts, proba_up, pred_up

# =========================
# Sentiment regime compute
# =========================
def _resample(df1m: pd.DataFrame, tf_min: int) -> pd.DataFrame:
    if df1m is None or df1m.empty or not isinstance(df1m.index, pd.DatetimeIndex):
        return pd.DataFrame()
    rule = f"{int(tf_min)}min"
    agg = {"open":"first","high":"max","low":"min","close":"last","volume":"sum"}
    try:
        bars = df1m.resample(rule, origin="start_day", label="right").agg(agg).dropna()
    except Exception:
        bars = pd.DataFrame()
    return bars

def _compute_sentiment() -> str:
    look_min = max(5, SENTIMENT_LOOKBACK_MIN)
    vals: List[float] = []
    for s in SENTIMENT_SYMBOLS:
        df = fetch_alpaca_1m(s, lookback_minutes=look_min*2)
        if df is None or df.empty:
            continue
        window = df.iloc[-look_min:]
        if len(window) < 2:
            continue
        vals.append(float(window["close"].iloc[-1]) / float(window["close"].iloc[0]) - 1.0)
    if not vals:
        return "neutral"
    avg = sum(vals) / len(vals)
    if avg >= SENTIMENT_NEUTRAL_BAND:
        return "bull"
    if avg <= -SENTIMENT_NEUTRAL_BAND:
        return "bear"
    return "neutral"

# =========================
# Signal construction
# =========================
def _build_signal_for_tf(symbol: str, df1m: pd.DataFrame, tf_min: int,
                         conf_threshold: float, r_multiple: float,
                         sentiment: str, shorts_enabled: bool) -> Optional[Dict[str, Any]]:
    bars = _resample(df1m, tf_min)
    if bars.empty:
        return None

    ts, proba_up, pred_up = _ml_features_and_pred(bars)
    if ts is None or proba_up is None:
        return None
    if not _in_session(ts.astimezone()):
        return None

    price = float(bars["close"].iloc[-1])
    if not np.isfinite(price) or price < MIN_PRICE:
        return None

    # risk unit: 1%; SL opposite, TP along trend scaled by r_multiple
    long_sl  = price * 0.99
    long_tp  = price * (1 + 0.01 * r_multiple)
    short_sl = price * 1.01
    short_tp = price * (1 - 0.01 * r_multiple)

    want_long  = (proba_up >= conf_threshold)
    want_short = ((1.0 - proba_up) >= conf_threshold) and shorts_enabled

    if USE_SENTIMENT_REGIME:
        if sentiment == "bull":
            want_short = False
        elif sentiment == "bear":
            want_long = False

    if want_long:
        qty = _position_qty(price, long_sl)
        if qty > 0:
            return {
                "symbol": symbol,
                "side": "buy",
                "qty": int(qty),
                "tp": float(round(long_tp, 4)),
                "sl": float(round(long_sl, 4)),
                "reason": "ml_pattern_long",
                "meta": {
                    "tf": f"{int(tf_min)}m",
                    "proba_up": proba_up,
                    "barTime": ts.astimezone(timezone.utc).isoformat()
                }
            }

    if want_short:
        qty = _position_qty(price, short_sl)
        if qty > 0:
            return {
                "symbol": symbol,
                "side": "sell",  # short open (runner handles permissions)
                "qty": int(qty),
                "tp": float(round(short_tp, 4)),
                "sl": float(round(short_sl, 4)),
                "reason": "ml_pattern_short",
                "meta": {
                    "tf": f"{int(tf_min)}m",
                    "proba_up": 1.0 - proba_up,
                    "barTime": ts.astimezone(timezone.utc).isoformat()
                }
            }

    return None

# =========================
# Adapter entrypoint
# =========================
def run_strategy(ctx) -> Iterable[Dict[str, Any]]:
    """
    ctx expectations (as provided by your runner):
      - ctx.universe: list[str] of symbols (already filtered by exchange if runner does that)
      - ctx.get_meta(symbol): may include exchange info (optional)
      - ctx.now: aware datetime
      - Optional: ctx.cache/ctx.memo if you support shared state
    We fetch bars via fetch_alpaca_1m() here.
    """
    if pd is None:
        return []

    # (Optional) filter by allowed exchanges if runner hasn’t already
    symbols = []
    for s in getattr(ctx, "universe", []):
        s = (s or "").strip().upper()
        if not s:
            continue
        if not ALLOWED_EXCHANGES:
            symbols.append(s)
            continue
        meta = {}
        try:
            meta = ctx.get_meta(s) or {}
        except Exception:
            meta = {}
        exch = str(meta.get("exchange") or meta.get("primary_exchange") or "").upper()
        if (not exch) or (exch in ALLOWED_EXCHANGES):
            symbols.append(s)

    # Sentiment once per loop
    sentiment = _compute_sentiment() if USE_SENTIMENT_REGIME else "neutral"

    intents: List[Dict[str, Any]] = []
    for sym in symbols:
        # Pull 1m bars from Alpaca
        df1m = fetch_alpaca_1m(sym, lookback_minutes=max(LOOKBACK_MIN, max(TF_MIN_LIST) * 240))
        if df1m is None or df1m.empty or ("close" not in df1m.columns):
            continue

        # Volume gate (today)
        try:
            today_mask = df1m.index.date == df1m.index[-1].date()
            todays_vol = float(df1m.loc[today_mask, "volume"].sum()) if today_mask.any() else 0.0
            if todays_vol < SCANNER_MIN_TODAY_VOL:
                continue
        except Exception:
            pass

        # Generate at most one intent per symbol (de-dupe), choose highest TF first
        produced = False
        for tf in sorted(TF_MIN_LIST, reverse=True):
            sig = _build_signal_for_tf(sym, df1m, tf, CONF_THR, R_MULT, sentiment, SHORTS_ENABLED)
            if not sig:
                continue

            # Optional de-dup per (symbol|side|barTime)
            if DEDUP_BY_SYMBOL:
                k = hashlib.sha256(f"{sym}|{sig['side']}|{sig['meta'].get('barTime','')}".encode()).hexdigest()
                # Use ctx.memo to persist de-dupe keys during the process lifetime
                memo = getattr(ctx, "memo", None)
                if memo is not None:
                    seen = memo.setdefault("ml_sentiment_seen", set())
                    if k in seen:
                        continue
                    seen.add(k)

            intents.append(sig)
            produced = True
            break  # don’t emit multiple TFs for same symbol this loop

    return intents
