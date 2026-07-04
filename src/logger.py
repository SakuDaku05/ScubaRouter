"""
Structured JSONL logger. Every record is both your day-3 calibration
data and, if there's time, your future classifier's training data.
"""
import json
import time
from pathlib import Path


class Logger:
    def __init__(self, log_path: str = "logs/run_log.jsonl"):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, record: dict):
        record = {"timestamp": time.time(), **record}
        with open(self.log_path, "a") as f:
            f.write(json.dumps(record) + "\n")
