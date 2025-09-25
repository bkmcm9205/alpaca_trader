# app/launch.py
import os, runpy, sys

# Ensure pandas_ta is available even if the package isn't installed
try:
    import pandas_ta  # noqa
except Exception:
    from app import ta_shim as _pta
    sys.modules["pandas_ta"] = _pta

role = (os.getenv("APP_ROLE") or "trader").strip().lower()
mod = "app.trainer_worker" if role == "trainer" else "app.strategy_runner"
print(f"[LAUNCH] role={role} module={mod}", flush=True)
runpy.run_module(mod, run_name="__main__")
