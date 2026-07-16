"""Per-rank Gaussian count timeline tracker.

학습 루프에서 rank 별 가우시안 카운트와 densify/OOM 이벤트를 기록하고
타일 종료/OOM 시 JSONL 로 flush. 시각화는 oom_viz.save_count_timeline_viz 가 처리.
"""

from __future__ import annotations

import atexit
import json
import os
from pathlib import Path
from typing import List, Optional


class GaussianCountTracker:
    def __init__(
        self,
        rank: int,
        tile_id: str,
        out_dir: str | os.PathLike,
        *,
        sample_every: int = 20,
    ):
        self.rank = int(rank)
        self.tile_id = str(tile_id)
        self.out_dir = Path(out_dir)
        self.sample_every = max(1, int(sample_every))
        self._records: List[dict] = []
        self._last_count: Optional[int] = None
        self._flushed = False
        atexit.register(self._atexit_flush)

    def maybe_record(self, iteration: int, count: int):
        if iteration <= 0:
            return
        if iteration % self.sample_every != 0:
            return
        self._records.append({"iter": int(iteration), "count": int(count), "event": ""})
        self._last_count = int(count)

    def record_event(self, iteration: int, count: int, event: str):
        self._records.append({"iter": int(iteration), "count": int(count), "event": str(event)})
        self._last_count = int(count)

    def flush(self) -> Optional[str]:
        if self._flushed:
            return None
        try:
            self.out_dir.mkdir(parents=True, exist_ok=True)
            path = self.out_dir / f"{self.tile_id}_rank{self.rank}.jsonl"
            with open(path, "w") as f:
                for r in self._records:
                    f.write(json.dumps(r) + "\n")
            self._flushed = True
            return str(path)
        except Exception as e:
            print(f"[count_tracker] rank {self.rank} flush failed: {e}", flush=True)
            return None

    def _atexit_flush(self):
        try:
            self.flush()
        except Exception:
            pass


_TRACKER: Optional[GaussianCountTracker] = None


def init_tracker(rank: int, tile_id: str, out_dir: str | os.PathLike, sample_every: int = 20):
    global _TRACKER
    _TRACKER = GaussianCountTracker(rank, tile_id, out_dir, sample_every=sample_every)
    return _TRACKER


def get_tracker() -> Optional[GaussianCountTracker]:
    return _TRACKER
