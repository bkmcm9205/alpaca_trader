# app/trainer.py
from __future__ import annotations
import os
import importlib
from datetime import datetime
from zoneinfo import ZoneInfo

# Map strategy name -> trainer module path
# Make sure this file exists and includes all the strategies you want trained.
# Example:
# TRAINERS = {
#   "trading_bot": "app.trainers.trading_bot_trainer",
#   "ml_merged": "app.trainers.ml_merged_trainer",
#   "ranked_ml": "app.trainers.ranked_ml_trainer",
#   "ml_sentiment": "app.trainers.ml_sentiment_trainer",
# }
try:
    from app.trainer_registry import TRAINERS
except Exception:
    TRAINERS = {}

ET_TZ = ZoneInfo("America/New_York")

def _now_hhmm_et() -> str:
    return datetime.now(ET_TZ).strftime("%H:%M")

def _in_window(start_hhmm: str, end_hhmm: str) -> bool:
    """Inclusive time window check in America/New_York."""
    try:
        now = _now_hhmm_et()
        return (start_hhmm <= now <= end_hhmm)
    except Exception:
        return True  # be permissive if parsing fails

def _log(msg: str):
    print(f"[TRAIN] {msg}", flush=True)

def _run_one(strategy: str) -> None:
    mod_path = TRAINERS.get(strategy)
    if not mod_path:
        _log(f"{strategy}: no trainer mapped (trainer_registry). Skipping.")
        return
    try:
        mod = importlib.import_module(mod_path)
    except Exception as e:
        _log(f"{strategy}: import error for '{mod_path}': {e}")
        return

    # Prefer train_and_evaluate(); fallback to evaluate_and_log()
    fn = getattr(mod, "train_and_evaluate", None)
    if fn is None:
        fn = getattr(mod, "evaluate_and_log", None)

    if fn is None or not callable(fn):
        _log(f"{strategy}: no entrypoint (train_and_evaluate/evaluate_and_log). Skipping.")
        return

    _log(f"{strategy}: running {mod_path}.{fn.__name__}()")
    try:
        fn()
        _log(f"{strategy}: completed.")
    except Exception as e:
        _log(f"{strategy}: ERROR -> {e}")

def main():
    # ---- schedule control ----
    start = os.getenv("TRAIN_RUN_START_ET", "19:00")
    end   = os.getenv("TRAIN_RUN_END_ET",   "20:00")
    force = os.getenv("FORCE_TRAIN", "0").lower() in ("1","true","yes")

    if not force and not _in_window(start, end):
        _log(f"Not in window {start}-{end} ET (now={_now_hhmm_et()}). Set FORCE_TRAIN=1 to override. Exiting.")
        return

    # ---- what to train ----
    raw = os.getenv("TRAIN_STRATEGIES", "")
    strategies = [s.strip() for s in raw.split(",") if s.strip()]
    if not strategies:
        _log("TRAIN_STRATEGIES is empty. Nothing to do.")
        return

    # ---- sanity: S3 envs helpful for most trainers ----
    bucket = os.getenv("S3_BUCKET") or ""
    region = os.getenv("S3_REGION") or ""
    if not bucket or not region:
        _log("WARNING: S3_BUCKET or S3_REGION not set. If your trainers write to S3, set them.")

    _log(f"Window={start}-{end} ET  Force={int(force)}  Strategies={strategies}")
    _log(f"S3_BUCKET={bucket or '(unset)'}  S3_REGION={region or '(unset)'}")

    # ---- run each strategy trainer ----
    for strat in strategies:
        _run_one(strat)

    _log("All requested trainers finished.")

if __name__ == "__main__":
    main()
