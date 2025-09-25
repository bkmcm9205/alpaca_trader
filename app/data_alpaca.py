from __future__ import annotations
import os, time
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Any, Optional

import requests
import pandas as pd

# ---- Config (from env) ----
ALPACA_DATA_BASE = os.getenv("ALPACA_DATA_BASE", "https://data.alpaca.markets").rstrip("/")
ALPACA_KEY       = os.getenv("ALPACA_KEY_ID", os.getenv("APCA_API_KEY_ID", ""))
ALPACA_SECRET    = os.getenv("ALPACA_SECRET_KEY", os.getenv("APCA_API_SECRET_KEY", ""))
ALPACA_FEED      = os.getenv("ALPACA_DATA_FEED", "sip")  # "sip" on Algo Trader Plus

def _auth_headers() -> Dict[str, str]:
    return {
        "Apca-Api-Key-Id": ALPACA_KEY,
        "Apca-Api-Secret-Key": ALPACA_SECRET,
        "accept": "application/json",
    }

def _bars_url(symbol: str) -> str:
    # Single-symbol endpoint; more robust & simpler to paginate
    return f"{ALPACA_DATA_BASE}/v2/stocks/{symbol}/bars"

def _to_df(bars: List[Dict[str, Any]]) -> pd.DataFrame:
    if not bars:
        return pd.DataFrame(columns=["open","high","low","close","volume"])
    # API returns ISO8601 timestamps in "t"
    idx = pd.to_datetime([b["t"] for b in bars], utc=True)
    df = pd.DataFrame({
        "open":   [float(b["o"]) for b in bars],
        "high":   [float(b["h"]) for b in bars],
        "low":    [float(b["l"]) for b in bars],
        "close":  [float(b["c"]) for b in bars],
        "volume": [int(b["v"])   for b in bars],
    }, index=idx)
    # Ensure strictly increasing index (sometimes last partial duplicates)
    df = df[~df.index.duplicated(keep="last")]
    df.sort_index(inplace=True)
    return df

def fetch_alpaca_1m(symbol: str, limit: int = 1500, end: Optional[datetime] = None) -> pd.DataFrame:
    """
    Fetch up to `limit` 1-minute bars for `symbol` from Alpaca, newest-first up to `end` (UTC).
    Handles pagination via next_page_token. Returns a tz-aware UTC-indexed DataFrame.
    """
    if not symbol:
        return pd.DataFrame()
    if not ALPACA_KEY or not ALPACA_SECRET:
        raise RuntimeError("Missing Alpaca credentials (ALPACA_KEY_ID / ALPACA_SECRET_KEY).")

    # We request using a start time so the API can paginate earlier bars deterministically.
    # Pad start by +20% minutes to be safe vs holidays/halts.
    end_utc = (end or datetime.utcnow().replace(tzinfo=timezone.utc))
    pad_min = int(limit * 1.2) + 10
    start_utc = end_utc - timedelta(minutes=max(limit, 1) + pad_min)

    gathered: List[Dict[str, Any]] = []
    page_token: Optional[str] = None

    remaining = max(limit, 1)
    tries = 0

    while remaining > 0 and tries < 50:  # hard safety stop
        tries += 1
        chunk = min(10000, remaining)  # API page size
        params = {
            "timeframe": "1Min",
            "start": start_utc.isoformat().replace("+00:00", "Z"),
            "end": end_utc.isoformat().replace("+00:00", "Z"),
            "limit": chunk,
            "feed": ALPACA_FEED,
        }
        if page_token:
            params["page_token"] = page_token

        r = requests.get(_bars_url(symbol), headers=_auth_headers(), params=params, timeout=60)
        if r.status_code == 429:
            # Rate limited – backoff and retry
            time.sleep(0.5)
            continue
        r.raise_for_status()
        data = r.json() or {}
        bars = data.get("bars", [])
        if not bars:
            break

        gathered.extend(bars)
        remaining -= len(bars)
        page_token = data.get("next_page_token")
        if not page_token:
            break
        # be gentle even on Algo Trader Plus
        time.sleep(0.05)

    # API returns most-recent-first? Normalize via sort in _to_df
    df = _to_df(gathered)
    if len(df) > limit:
        df = df.iloc[-limit:]  # keep the most recent `limit` rows
    return df

# Optional convenience for a quick historical backfill (date-range)
def fetch_alpaca_1m_range(symbol: str, start: datetime, end: datetime | None = None) -> pd.DataFrame:
    """
    Fetch 1m bars for a date range [start, end]. Uses repeated calls to fetch_alpaca_1m.
    """
    end = end or datetime.utcnow().replace(tzinfo=timezone.utc)
    out = []
    cursor = end
    while cursor > start:
        df = fetch_alpaca_1m(symbol, limit=10000, end=cursor)
        if df is None or df.empty:
            break
        out.append(df)
        # step back by len(df) minutes
        cursor = df.index[0].to_pydatetime()
        # guard
        if len(out) > 30:
            break
    if not out:
        return pd.DataFrame()
    out_df = pd.concat(out).sort_index()
    return out_df[(out_df.index >= pd.to_datetime(start, utc=True)) & (out_df.index <= pd.to_datetime(end, utc=True))]
