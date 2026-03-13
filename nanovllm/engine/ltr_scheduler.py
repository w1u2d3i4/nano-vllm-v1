import os
import time
from collections import deque

import joblib
import numpy as np
from sklearn.linear_model import SGDRegressor

from nanovllm.config import Config
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.sequence import Sequence
from nanovllm.utils.data_collector import DataCollector


class LTRScheduler(Scheduler):
    """
    Learning-to-Rank scheduler: Shortest-Job-First via predicted output length.

    Three phases:
      FCFS       (< HEURISTIC_THRESHOLD samples): normal FIFO
      Heuristic  (< MODEL_THRESHOLD):             sort waiting by prompt_length
      Model      (>= MODEL_THRESHOLD):            sort waiting by predicted output_length
    """

    PHASE_FCFS = 0
    PHASE_HEURISTIC = 1
    PHASE_MODEL = 2

    HEURISTIC_THRESHOLD = 50
    MODEL_THRESHOLD = 200
    STARVATION_LIMIT = 30.0
    UPDATE_INTERVAL = 100
    REORDER_INTERVAL = 100  # Reorder every N new requests

    def __init__(self, config: Config):
        super().__init__(config)
        self.collector = DataCollector(maxlen=50000)
        self.model: SGDRegressor | None = None
        self.phase = self.PHASE_FCFS
        self.frozen = False  # Set True to disable all online learning overhead
        self.last_train_size = 0
        self.ltr_data_path = config.ltr_data_path
        self.update_log: list[dict] = []
        self._need_reorder = False
        self._new_requests_since_reorder = 0
        self._earliest_arrival_time = float('inf')  # Track earliest arrival for starvation check
        # Performance tracking
        self._predict_time_total = 0.0
        self._predict_count = 0
        self._reorder_count = 0

        if self.ltr_data_path:
            model_file = self.ltr_data_path + ".model"
            if os.path.exists(model_file):
                self.model = joblib.load(model_file)
                self.collector.load_from_file(self.ltr_data_path)
                self.last_train_size = self.collector.size
                self._update_phase()
                print(f"[LTR] Warm start: loaded model + {self.collector.size} samples, phase={self._phase_name}")
            elif os.path.exists(self.ltr_data_path):
                self.collector.load_from_file(self.ltr_data_path)
                self._maybe_train(force=True)
                self._update_phase()
                print(f"[LTR] Cold start: loaded {self.collector.size} samples, phase={self._phase_name}")

    @property
    def _phase_name(self) -> str:
        return ["FCFS", "Heuristic", "Model"][self.phase]

    def add(self, seq: Sequence):
        super().add(seq)
        self._new_requests_since_reorder += 1
        if self._new_requests_since_reorder >= self.REORDER_INTERVAL:
            self._need_reorder = True
        # Track earliest arrival time for starvation check optimization
        if seq.arrival_time < self._earliest_arrival_time:
            self._earliest_arrival_time = seq.arrival_time

    # ── scheduling ────────────────────────────────────────────

    def schedule(self) -> list[Sequence]:
        if not self.frozen:
            self._update_phase()

        if self._need_reorder and self.phase > self.PHASE_FCFS and len(self.waiting) > 1:
            self._reorder_waiting()
            self._need_reorder = False
            self._new_requests_since_reorder = 0

        return super().schedule()

    def postprocess(self, seqs, token_ids, seq_need_compute_logits):
        if self.frozen:
            super().postprocess(seqs, token_ids, seq_need_compute_logits)
            return

        finished_before = {s.seq_id for s in seqs if s.is_finished}
        super().postprocess(seqs, token_ids, seq_need_compute_logits)

        for seq in seqs:
            if seq.is_finished and seq.seq_id not in finished_before:
                self.collector.collect(seq.num_prompt_tokens, seq.num_completion_tokens)

        self._maybe_train()

    # ── phase management ──────────────────────────────────────

    def _update_phase(self):
        n = self.collector.size
        if n >= self.MODEL_THRESHOLD and self.model is not None:
            new_phase = self.PHASE_MODEL
        elif n >= self.HEURISTIC_THRESHOLD:
            new_phase = self.PHASE_HEURISTIC
        else:
            new_phase = self.PHASE_FCFS
        if new_phase != self.phase:
            self.phase = new_phase
            print(f"[LTR] Phase → {self._phase_name} ({n} samples)")

    # ── waiting-queue reordering (SJF) ────────────────────────

    def _reorder_waiting(self):
        self._reorder_count += 1
        seqs = list(self.waiting)
        now = time.time()

        if self.phase == self.PHASE_HEURISTIC:
            seqs.sort(key=lambda s: s.num_prompt_tokens)
        elif self.phase == self.PHASE_MODEL and self.model is not None:
            t_start = time.time()
            preds = self._predict_batch(seqs)
            self._predict_time_total += (time.time() - t_start)
            self._predict_count += 1
            seqs.sort(key=lambda s: preds.get(s.seq_id, float("inf")))

        # Starvation check: only scan if earliest request might be starved
        if now - self._earliest_arrival_time > self.STARVATION_LIMIT:
            starved, rest = [], []
            new_earliest = float('inf')
            for s in seqs:
                if now - s.arrival_time > self.STARVATION_LIMIT:
                    starved.append(s)
                else:
                    rest.append(s)
                    if s.arrival_time < new_earliest:
                        new_earliest = s.arrival_time
            self.waiting = deque(starved + rest)
            self._earliest_arrival_time = new_earliest if rest else float('inf')
        else:
            # No starvation possible, just use sorted order
            self.waiting = deque(seqs)

    def _predict_batch(self, seqs: list[Sequence]) -> dict[int, float]:
        if self.model is None:
            return {}
        X = np.array([[s.num_prompt_tokens] for s in seqs], dtype=np.float64)
        preds = self.model.predict(X)
        return {s.seq_id: float(p) for s, p in zip(seqs, preds)}

    # ── model training / online learning ──────────────────────

    def _maybe_train(self, force: bool = False):
        n = self.collector.size
        if not force and n - self.last_train_size < self.UPDATE_INTERVAL:
            return
        if n < self.MODEL_THRESHOLD:
            return

        X, y = self.collector.get_training_data()
        if X is None:
            return

        if self.model is None:
            self.model = SGDRegressor(max_iter=1000, tol=1e-3)
            self.model.fit(X, y)
        else:
            X_r, y_r = self.collector.get_recent(self.UPDATE_INTERVAL)
            if X_r is not None:
                self.model.partial_fit(X_r, y_r)

        self.last_train_size = n
        self._need_reorder = True
        self._new_requests_since_reorder = 0  # Reset counter after model update
        self.update_log.append({
            "time": time.time(),
            "samples": n,
            "phase": self._phase_name,
        })

    # ── persistence ───────────────────────────────────────────

    def save_state(self):
        if not self.ltr_data_path:
            return
        self.collector.save_to_file(self.ltr_data_path)
        if self.model is not None:
            joblib.dump(self.model, self.ltr_data_path + ".model")
        print(f"[LTR] Saved: {self.collector.size} samples + model to {self.ltr_data_path}")

        # Print performance statistics
        if self._predict_count > 0:
            avg_predict_time = self._predict_time_total / self._predict_count
            print(f"[LTR-PERF] Reorder count: {self._reorder_count}")
            print(f"[LTR-PERF] Predict count: {self._predict_count}")
            print(f"[LTR-PERF] Total predict time: {self._predict_time_total:.3f}s")
            print(f"[LTR-PERF] Avg predict time: {avg_predict_time*1000:.2f}ms")
            print(f"[LTR-PERF] Predict overhead: {self._predict_time_total:.1f}s ({self._predict_time_total/self._reorder_count*1000:.2f}ms per reorder)")
