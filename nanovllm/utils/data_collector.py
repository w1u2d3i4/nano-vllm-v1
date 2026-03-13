import json
from collections import deque

import numpy as np


class DataCollector:
    """Ring buffer for LTR training samples (prompt_length → output_length)."""

    def __init__(self, maxlen: int = 50000):
        self.buffer: deque[tuple[int, int]] = deque(maxlen=maxlen)

    @property
    def size(self) -> int:
        return len(self.buffer)

    def collect(self, prompt_length: int, output_length: int):
        self.buffer.append((prompt_length, output_length))

    def load_from_file(self, path: str):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                self.buffer.append((d["prompt_length"], d["output_length"]))

    def save_to_file(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            for prompt_len, output_len in self.buffer:
                f.write(json.dumps({"prompt_length": prompt_len, "output_length": output_len}) + "\n")

    def get_training_data(self) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
        if not self.buffer:
            return None, None
        data = list(self.buffer)
        X = np.array([[d[0]] for d in data], dtype=np.float64)
        y = np.array([d[1] for d in data], dtype=np.float64)
        return X, y

    def get_recent(self, n: int) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
        if not self.buffer:
            return None, None
        data = list(self.buffer)[-n:]
        X = np.array([[d[0]] for d in data], dtype=np.float64)
        y = np.array([d[1] for d in data], dtype=np.float64)
        return X, y
