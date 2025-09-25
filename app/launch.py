# app/launch.py
import os, runpy

role = (os.getenv("APP_ROLE") or "trader").strip().lower()
mod = "app.trainer_worker" if role == "trainer" else "app.strategy_runner"
print(f"[LAUNCH] role={role} module={mod}", flush=True)
runpy.run_module(mod, run_name="__main__")
