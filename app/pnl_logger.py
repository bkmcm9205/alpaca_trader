# app/pnl_logger.py
import os, io, json, csv, boto3, time
from datetime import datetime, timezone
from typing import Dict, Any, List

S3_BUCKET = os.getenv("S3_BUCKET")
TRADES_PREFIX = os.getenv("TRADES_LOG_S3_PREFIX", "logs/trades/")
LEDGER_PREFIX = os.getenv("LEDGER_S3_PREFIX", "ledgers/")

s3 = boto3.client("s3")

def _today_date():
    return datetime.now(timezone.utc).astimezone().date().isoformat()

def _s3_key_trades():
    return f"{TRADES_PREFIX}{_today_date()}.csv"

def _s3_key_ledger(strategy_id: str):
    return f"{LEDGER_PREFIX}{strategy_id}/{_today_date()}.json"

def _s3_get_json_or_empty(bucket, key):
    try:
        body = s3.get_object(Bucket=bucket, Key=key)["Body"].read()
        return json.loads(body)
    except s3.exceptions.NoSuchKey:
        return {}
    except Exception:
        return {}

def _s3_put_json(bucket, key, obj):
    s3.put_object(Bucket=bucket, Key=key, Body=json.dumps(obj).encode("utf-8"))

def append_trades(rows: List[Dict[str, Any]]):
    """
    rows columns (all strings ok):
    ts_utc,symbol,side,qty,avg_fill,client_order_id,strategy,tag,realized_pnl,unrealized_pnl,account_equity
    """
    key = _s3_key_trades()
    header = ["ts_utc","symbol","side","qty","avg_fill","client_order_id",
              "strategy","tag","realized_pnl","unrealized_pnl","account_equity"]
    # read existing (if any)
    try:
        obj = s3.get_object(Bucket=S3_BUCKET, Key=key)["Body"].read().decode("utf-8")
        buf = io.StringIO(obj)
        existing = list(csv.reader(buf))
        needs_header = (len(existing) == 0)
        out = io.StringIO()
        w = csv.writer(out, lineterminator="\n")
        if needs_header:
            w.writerow(header)
        # re-append existing
        for r in existing:
            w.writerow(r)
    except s3.exceptions.NoSuchKey:
        out = io.StringIO()
        w = csv.writer(out, lineterminator="\n")
        w.writerow(header)

    # append new
    for r in rows:
        w.writerow([r.get(k, "") for k in header])

    s3.put_object(Bucket=S3_BUCKET, Key=key, Body=out.getvalue().encode("utf-8"))

def update_ledger_with_fills(strategy_id: str, fills: List[Dict[str, Any]]):
    """
    Maintain per-strategy net_qty/avg_cost by symbol from fills that match our tag/strategy_id.
    """
    key = _s3_key_ledger(strategy_id)
    ledger = _s3_get_json_or_empty(S3_BUCKET, key) or {"symbols": {}, "baseline_equity": None}

    for f in fills:
        sym = f.get("symbol")
        side = f.get("side")  # 'buy' or 'sell'
        qty = float(f.get("qty", 0))
        price = float(f.get("price", 0))
        symrec = ledger["symbols"].setdefault(sym, {"net_qty": 0.0, "avg_cost": 0.0})
        if side == "buy":
            # weighted average
            new_qty = symrec["net_qty"] + qty
            if new_qty > 0:
                symrec["avg_cost"] = (symrec["avg_cost"] * symrec["net_qty"] + price * qty) / new_qty
            symrec["net_qty"] = new_qty
        else:
            # sell reduces qty; realized pnl tracked elsewhere if needed
            symrec["net_qty"] = symrec["net_qty"] - qty
            if symrec["net_qty"] <= 0:
                symrec["avg_cost"] = 0.0

    _s3_put_json(S3_BUCKET, key, ledger)

def compute_virtual_equity(strategy_id: str, account_equity_baseline: float,
                           prices: Dict[str, float]) -> Dict[str, float]:
    """
    Returns dict with: baseline, mtm, virtual_equity
    """
    key = _s3_key_ledger(strategy_id)
    ledger = _s3_get_json_or_empty(S3_BUCKET, key) or {"symbols": {}, "baseline_equity": None}
    baseline = ledger.get("baseline_equity")
    if baseline is None:
        # first call today: set baseline to provided account baseline
        ledger["baseline_equity"] = account_equity_baseline
        _s3_put_json(S3_BUCKET, key, ledger)
        baseline = account_equity_baseline

    mtm = 0.0
    for sym, rec in ledger["symbols"].items():
        px = float(prices.get(sym, 0.0) or 0.0)
        mtm += (px - rec.get("avg_cost", 0.0)) * rec.get("net_qty", 0.0)

    return {"baseline": baseline, "mtm": mtm, "virtual_equity": baseline + mtm}
