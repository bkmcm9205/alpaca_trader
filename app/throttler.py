import time
from collections import deque

class PerMinuteLimiter:
    """Simple rolling one-minute limiter."""
    def __init__(self, max_per_minute: int):
        self.max = max_per_minute
        self.ts = deque()

    def allow(self) -> bool:
        now = time.time()
        # drop older than 60s
        while self.ts and now - self.ts[0] > 60.0:
            self.ts.popleft()
        if len(self.ts) < self.max:
            self.ts.append(now)
            return True
        return False

    def wait_until_allowed(self):
        while not self.allow():
            time.sleep(0.25)
