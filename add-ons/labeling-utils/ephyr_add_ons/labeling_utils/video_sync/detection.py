"""TTL / flash detection, train clustering, and sequence matching for video sync."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence, Tuple

import numpy as np
from scipy.signal import find_peaks

EDGE_RISING = "rising"
EDGE_FALLING = "falling"

# Re-export compatible TTL detector (same logic as events_detection).
def detect_ttl_edges(
    signal_1d: np.ndarray,
    threshold: float,
    edge: str,
    min_distance_samples: int = 1,
) -> np.ndarray:
    """Return sample indices of TTL threshold crossings."""
    x = np.asarray(signal_1d, dtype=np.float64)
    if x.size < 2:
        return np.asarray([], dtype=np.int64)
    prev = x[:-1]
    curr = x[1:]
    if edge == EDGE_FALLING:
        crossings = np.flatnonzero((prev >= threshold) & (curr < threshold)) + 1
    else:
        crossings = np.flatnonzero((prev < threshold) & (curr >= threshold)) + 1
    if crossings.size == 0:
        return crossings.astype(np.int64)
    distance = max(1, int(min_distance_samples))
    if distance <= 1:
        return crossings.astype(np.int64)
    kept = [int(crossings[0])]
    for idx in crossings[1:]:
        if int(idx) - kept[-1] >= distance:
            kept.append(int(idx))
    return np.asarray(kept, dtype=np.int64)


@dataclass
class TtlTrain:
    """A temporally clustered sequence of TTL pulse times (ms)."""

    times_ms: np.ndarray
    train_id: int = 0

    @property
    def median_ipi_ms(self) -> float:
        t = np.asarray(self.times_ms, dtype=np.float64)
        if t.size < 2:
            return 0.0
        return float(np.median(np.diff(t)))

    @property
    def frequency_hz(self) -> float:
        ipi = self.median_ipi_ms
        if ipi <= 0:
            return 0.0
        return 1000.0 / ipi

    def label(self) -> str:
        n = int(np.asarray(self.times_ms).size)
        freq = self.frequency_hz
        t0 = float(self.times_ms[0]) if n else 0.0
        t1 = float(self.times_ms[-1]) if n else 0.0
        if freq > 0:
            return f"Train {self.train_id}: {n} pulses @ {freq:.2f} Hz [{t0:.0f}–{t1:.0f} ms]"
        return f"Train {self.train_id}: {n} pulses [{t0:.0f}–{t1:.0f} ms]"


@dataclass
class MatchCandidate:
    train_id: int
    score: float
    pairs: List[Tuple[float, float]] = field(default_factory=list)


SOURCE_VIDEO = "video"
SOURCE_NWB = "nwb"


@dataclass
class VideoMeta:
    path: str
    fps: float
    frame_count: int
    duration_ms: float
    source_kind: str = SOURCE_VIDEO
    series_location: str = ""

    @property
    def is_nwb(self) -> bool:
        return str(self.source_kind) == SOURCE_NWB


class VideoFileFrameSource:
    """Read frames from a video file via OpenCV."""

    def __init__(self, path: str):
        self.path = str(path)

    def read_bgr(self, frame_index: int):
        return read_frame_bgr(self.path, frame_index)

    def close(self) -> None:
        return None


def open_video_meta(path: str) -> VideoMeta:
    import cv2

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    if fps <= 0:
        fps = 30.0
    duration_ms = (frame_count / fps) * 1000.0 if frame_count > 0 else 0.0
    return VideoMeta(
        path=str(path),
        fps=fps,
        frame_count=frame_count,
        duration_ms=duration_ms,
        source_kind=SOURCE_VIDEO,
    )


def samples_to_ms(sample_indices: np.ndarray, sample_rate: float) -> np.ndarray:
    if sample_rate <= 0:
        return np.asarray([], dtype=np.float64)
    return (np.asarray(sample_indices, dtype=np.float64) * 1000.0) / float(sample_rate)


def cluster_ttl_trains(
    times_ms: Sequence[float],
    gap_factor: float = 4.0,
    min_gap_ms: float = 500.0,
) -> List[TtlTrain]:
    """Split TTL pulse times into trains using large IPI gaps."""
    t = np.asarray(sorted(float(x) for x in times_ms), dtype=np.float64)
    if t.size == 0:
        return []
    if t.size == 1:
        return [TtlTrain(times_ms=t, train_id=0)]

    ipis = np.diff(t)
    median_ipi = float(np.median(ipis)) if ipis.size else 0.0
    threshold = max(float(min_gap_ms), float(gap_factor) * median_ipi) if median_ipi > 0 else float(min_gap_ms)

    trains: List[TtlTrain] = []
    start = 0
    for i, gap in enumerate(ipis):
        if float(gap) > threshold:
            chunk = t[start : i + 1]
            trains.append(TtlTrain(times_ms=chunk.copy(), train_id=len(trains)))
            start = i + 1
    trains.append(TtlTrain(times_ms=t[start:].copy(), train_id=len(trains)))
    return trains


def _ipi_pattern(times_ms: np.ndarray) -> np.ndarray:
    t = np.asarray(times_ms, dtype=np.float64)
    if t.size < 2:
        return np.asarray([], dtype=np.float64)
    ipi = np.diff(t)
    med = float(np.median(ipi))
    if med <= 0:
        return ipi
    return ipi / med


def _pattern_score(a: np.ndarray, b: np.ndarray) -> float:
    """Score in [0, 1] comparing relative IPI patterns (higher is better)."""
    if a.size == 0 or b.size == 0:
        return 0.0
    n = min(int(a.size), int(b.size))
    if n < 1:
        return 0.0
    aa = a[:n]
    bb = b[:n]
    denom = np.maximum(np.maximum(np.abs(aa), np.abs(bb)), 1e-9)
    rel_err = np.abs(aa - bb) / denom
    return float(np.clip(1.0 - float(np.mean(rel_err)), 0.0, 1.0))


def align_sequences(
    signal_times_ms: Sequence[float],
    video_times_ms: Sequence[float],
    max_skip: int = 2,
) -> List[Tuple[float, float]]:
    """Greedy 1:1 alignment of two sorted time sequences allowing limited skips."""
    s = [float(x) for x in signal_times_ms]
    v = [float(x) for x in video_times_ms]
    if not s or not v:
        return []

    pairs: List[Tuple[float, float]] = []
    i = 0
    j = 0
    # Anchor first pulse of each sequence that we can pair.
    while i < len(s) and j < len(v):
        # Prefer advancing both; allow skipping the side that is "ahead" in relative progress.
        pairs.append((s[i], v[j]))
        i += 1
        j += 1
        if i >= len(s) or j >= len(v):
            break
        # Look-ahead: if next signal IPI is much larger, maybe flash was missed (skip flash)
        # or vice versa.
        skips = 0
        while i < len(s) and j < len(v) and skips < max_skip:
            ds = s[i] - s[i - 1]
            dv = v[j] - v[j - 1]
            if ds <= 0 or dv <= 0:
                break
            ratio = ds / dv
            if ratio > 1.6 and j + 1 < len(v):
                # flash IPI too short relative to signal → skip a flash
                j += 1
                skips += 1
                continue
            if ratio < 0.625 and i + 1 < len(s):
                i += 1
                skips += 1
                continue
            break
    return pairs


def match_trains_to_flashes(
    trains: Sequence[TtlTrain],
    flash_times_ms: Sequence[float],
    min_score: float = 0.35,
) -> List[MatchCandidate]:
    """Rank TTL trains by how well their IPI pattern matches flash times."""
    flashes = np.asarray(sorted(float(x) for x in flash_times_ms), dtype=np.float64)
    flash_pat = _ipi_pattern(flashes)
    candidates: List[MatchCandidate] = []
    for train in trains:
        t = np.asarray(train.times_ms, dtype=np.float64)
        score = _pattern_score(_ipi_pattern(t), flash_pat)
        # Bonus if counts are similar
        if t.size > 0 and flashes.size > 0:
            count_ratio = min(t.size, flashes.size) / max(t.size, flashes.size)
            score = 0.7 * score + 0.3 * float(count_ratio)
        if score < min_score and len(trains) > 1:
            continue
        pairs = align_sequences(t.tolist(), flashes.tolist())
        candidates.append(MatchCandidate(train_id=int(train.train_id), score=float(score), pairs=pairs))
    candidates.sort(key=lambda c: c.score, reverse=True)
    if not candidates and trains and flashes.size > 0:
        # Fallback: best train by count proximity
        best = min(trains, key=lambda tr: abs(int(tr.times_ms.size) - int(flashes.size)))
        pairs = align_sequences(best.times_ms.tolist(), flashes.tolist())
        candidates.append(MatchCandidate(train_id=int(best.train_id), score=0.0, pairs=pairs))
    return candidates


def detect_flashes_in_video(
    path: str,
    sensitivity: float = 2.5,
    min_distance_ms: float = 50.0,
    sample_fps: float = 0.0,
    progress_callback=None,
) -> Tuple[np.ndarray, VideoMeta]:
    """Detect LED/brightness flashes; return flash times in video ms and meta.

    ``sensitivity`` is the peak prominence factor relative to MAD of Δluminance.
    """
    import cv2

    meta = open_video_meta(path)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")

    fps = meta.fps
    target_fps = float(sample_fps) if sample_fps and sample_fps > 0 else min(fps, 30.0)
    step = max(1, int(round(fps / target_fps))) if target_fps > 0 else 1
    effective_fps = fps / step

    luminances: List[float] = []
    frame_indices: List[int] = []
    total = max(1, meta.frame_count)
    idx = 0
    while True:
        ok = cap.grab()
        if not ok:
            break
        if idx % step == 0:
            ok2, frame = cap.retrieve()
            if not ok2 or frame is None:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            luminances.append(float(np.mean(gray)))
            frame_indices.append(idx)
            if progress_callback is not None and idx % (step * 30) == 0:
                progress_callback(min(99, int(100 * idx / total)))
        idx += 1
    cap.release()

    if len(luminances) < 3:
        return np.asarray([], dtype=np.float64), meta

    lum = np.asarray(luminances, dtype=np.float64)
    delta = np.diff(lum, prepend=lum[0])
    # Positive brightness jumps
    med = float(np.median(delta))
    mad = float(np.median(np.abs(delta - med))) or 1.0
    height = med + float(sensitivity) * 1.4826 * mad
    min_dist_samples = max(1, int((min_distance_ms / 1000.0) * effective_fps))
    peaks, _ = find_peaks(delta, height=height, distance=min_dist_samples)

    flash_times = []
    for p in peaks:
        frame_i = frame_indices[int(p)]
        flash_times.append((frame_i / fps) * 1000.0)
    return np.asarray(flash_times, dtype=np.float64), meta


def read_frame_bgr(path: str, frame_index: int):
    """Read a single BGR frame; returns None on failure."""
    import cv2

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return None
    frame_index = max(0, int(frame_index))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return None
    return frame
