"""Affine / offset mapping from signal time to video time."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

MAPPING_FILENAME = "video_sync_mapping.json"


@dataclass
class TimeMapping:
    """Map signal milliseconds to video milliseconds via offset or affine fit.

    One pair → ``video_ms = signal_ms + b``.
    Two or more pairs → least-squares ``video_ms = a * signal_ms + b``.
    """

    pairs: List[Tuple[float, float]] = field(default_factory=list)
    a: float = 1.0
    b: float = 0.0

    def __post_init__(self) -> None:
        self.refit()

    @property
    def trusted_interval_signal_ms(self) -> Optional[Tuple[float, float]]:
        if not self.pairs:
            return None
        sig = [p[0] for p in self.pairs]
        return float(min(sig)), float(max(sig))

    def is_trusted(self, signal_ms: float) -> bool:
        interval = self.trusted_interval_signal_ms
        if interval is None:
            return False
        lo, hi = interval
        return lo <= float(signal_ms) <= hi

    def refit(self) -> None:
        if not self.pairs:
            self.a = 1.0
            self.b = 0.0
            return
        xs = np.asarray([p[0] for p in self.pairs], dtype=np.float64)
        ys = np.asarray([p[1] for p in self.pairs], dtype=np.float64)
        if xs.size == 1:
            self.a = 1.0
            self.b = float(ys[0] - xs[0])
            return
        # video = a * signal + b
        A = np.column_stack([xs, np.ones_like(xs)])
        try:
            coef, *_ = np.linalg.lstsq(A, ys, rcond=None)
            self.a = float(coef[0])
            self.b = float(coef[1])
            if not np.isfinite(self.a) or abs(self.a) < 1e-12:
                self.a = 1.0
                self.b = float(np.mean(ys - xs))
        except Exception:
            self.a = 1.0
            self.b = float(np.mean(ys - xs))

    def set_pairs(self, pairs: Sequence[Tuple[float, float]]) -> None:
        self.pairs = [(float(s), float(v)) for s, v in pairs]
        self.refit()

    def add_pair(self, signal_ms: float, video_ms: float) -> None:
        self.pairs.append((float(signal_ms), float(video_ms)))
        self.refit()

    def remove_pair_at(self, index: int) -> None:
        if 0 <= index < len(self.pairs):
            del self.pairs[index]
            self.refit()

    def to_video_ms(self, signal_ms: float) -> float:
        return float(self.a * float(signal_ms) + self.b)

    def to_signal_ms(self, video_ms: float) -> float:
        if abs(self.a) < 1e-12:
            return float(video_ms - self.b)
        return float((float(video_ms) - self.b) / self.a)

    def to_frame_index(self, signal_ms: float, fps: float) -> int:
        video_ms = self.to_video_ms(signal_ms)
        if fps <= 0:
            return 0
        return int(round((video_ms / 1000.0) * float(fps)))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pairs": [[float(s), float(v)] for s, v in self.pairs],
            "a": float(self.a),
            "b": float(self.b),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TimeMapping":
        raw_pairs = data.get("pairs") or []
        pairs = [(float(p[0]), float(p[1])) for p in raw_pairs if len(p) >= 2]
        mapping = cls(pairs=pairs)
        return mapping

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> Optional["TimeMapping"]:
        path = Path(path)
        if not path.is_file():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                return None
            return cls.from_dict(data)
        except Exception:
            return None


def mapping_path(add_on_data_dir: Path) -> Path:
    return Path(add_on_data_dir) / MAPPING_FILENAME
