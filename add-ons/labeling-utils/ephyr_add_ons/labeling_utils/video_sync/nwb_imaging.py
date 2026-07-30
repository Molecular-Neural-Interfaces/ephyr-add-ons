# Copyright (C) 2026 Life Improvement by Future Technologies (LIFT)
# SPDX-License-Identifier: GPL-3.0-only

"""NWB optical imaging series discovery and frame reading for video sync."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np

from ephyr_add_ons.labeling_utils.video_sync.detection import SOURCE_NWB, VideoMeta
from ephyr_add_ons.labeling_utils.video_sync.mapping import TimeMapping


@dataclass(frozen=True)
class NwbImagingSeriesInfo:
    location: str
    name: str
    n_frames: int
    height: int
    width: int
    fps: float
    duration_ms: float
    starting_time_s: float
    unit: str


def _series_rate_and_start(series) -> Tuple[Optional[float], float]:
    rate = getattr(series, "rate", None)
    starting = getattr(series, "starting_time", None)
    start_s = float(starting) if starting is not None else 0.0
    if rate is not None:
        rate_f = float(rate)
        if rate_f > 0:
            return rate_f, start_s
    timestamps = getattr(series, "timestamps", None)
    if timestamps is None:
        return None, start_s
    try:
        n = len(timestamps)
        if n < 2:
            return None, float(timestamps[0]) if n else start_s
        first = float(timestamps[0])
        last = float(timestamps[n - 1])
        if last <= first:
            return None, first
        return float((n - 1) / (last - first)), first
    except Exception:
        return None, start_s


def _is_imaging_series(obj) -> bool:
    try:
        from pynwb.image import ImageSeries
        from pynwb.ophys import OnePhotonSeries, TwoPhotonSeries
    except Exception:
        return False
    return isinstance(obj, (ImageSeries, OnePhotonSeries, TwoPhotonSeries))


def _iter_containers(nwbfile) -> Iterable[Tuple[str, object]]:
    for name, obj in nwbfile.acquisition.items():
        yield from _iter_from_obj(f"acquisition/{name}", obj)
    for module_name, module in nwbfile.processing.items():
        yield from _iter_from_obj(f"processing/{module_name}", module)


def _iter_from_obj(location: str, obj) -> Iterable[Tuple[str, object]]:
    if _is_imaging_series(obj):
        yield location, obj
        return
    for attr in ("data_interfaces", "time_series", "electrical_series"):
        children = getattr(obj, attr, None)
        if not children:
            continue
        if hasattr(children, "items"):
            iterable = children.items()
        else:
            iterable = enumerate(children)
        for name, child in iterable:
            yield from _iter_from_obj(f"{location}/{name}", child)


def _series_info(location: str, series) -> Optional[NwbImagingSeriesInfo]:
    data = getattr(series, "data", None)
    shape = getattr(data, "shape", None)
    if not shape or len(shape) < 2:
        return None
    # Common layouts: (frames, H, W) or (frames, H, W, C)
    if len(shape) == 2:
        # Ambiguous; treat as (frames, pixels) — skip for display
        return None
    n_frames = int(shape[0])
    height = int(shape[1])
    width = int(shape[2]) if len(shape) >= 3 else int(shape[1])
    if n_frames <= 0 or height <= 0 or width <= 0:
        return None
    rate, start_s = _series_rate_and_start(series)
    if rate is None or rate <= 0:
        return None
    duration_ms = (n_frames / rate) * 1000.0
    return NwbImagingSeriesInfo(
        location=location,
        name=str(getattr(series, "name", "") or location.rsplit("/", 1)[-1]),
        n_frames=n_frames,
        height=height,
        width=width,
        fps=float(rate),
        duration_ms=float(duration_ms),
        starting_time_s=float(start_s),
        unit=str(getattr(series, "unit", "") or ""),
    )


def list_imaging_series(nwb_path: Path) -> List[NwbImagingSeriesInfo]:
    from pynwb import NWBHDF5IO

    path = Path(nwb_path)
    with NWBHDF5IO(str(path), "r", load_namespaces=True) as io:
        nwbfile = io.read()
        found: List[NwbImagingSeriesInfo] = []
        for location, series in _iter_containers(nwbfile):
            info = _series_info(location, series)
            if info is not None:
                found.append(info)
    found.sort(key=lambda item: item.n_frames * item.height * item.width, reverse=True)
    return found


def _find_series(nwbfile, location: str):
    for candidate_location, series in _iter_containers(nwbfile):
        if candidate_location == location:
            return series
    raise ValueError(f"NWB imaging series not found: {location}")


def electrical_starting_time_s(nwb_path: Path, preferred_location: str = "") -> Optional[float]:
    """Best-effort ephys timeline origin (seconds) for aligning imaging."""
    from pynwb import NWBHDF5IO, TimeSeries
    from pynwb.ecephys import ElectricalSeries

    path = Path(nwb_path)
    with NWBHDF5IO(str(path), "r", load_namespaces=True) as io:
        nwbfile = io.read()
        preferred = preferred_location.strip()
        if preferred:
            # Direct walk of acquisition/processing for any TimeSeries at location.
            for location, obj in _iter_any_timeseries(nwbfile):
                if location == preferred and isinstance(obj, TimeSeries):
                    _rate, start_s = _series_rate_and_start(obj)
                    return float(start_s)

        # Prefer largest ElectricalSeries
        best = None
        best_size = -1
        for _location, obj in _iter_any_timeseries(nwbfile):
            if not isinstance(obj, ElectricalSeries):
                continue
            shape = getattr(getattr(obj, "data", None), "shape", None)
            size = int(np.prod(shape)) if shape else 0
            if size > best_size:
                best_size = size
                best = obj
        if best is None:
            return None
        _rate, start_s = _series_rate_and_start(best)
        return float(start_s)


def _iter_any_timeseries(nwbfile) -> Iterable[Tuple[str, object]]:
    from pynwb import TimeSeries

    def walk(location: str, obj):
        if isinstance(obj, TimeSeries):
            yield location, obj
            return
        for attr in ("data_interfaces", "time_series", "electrical_series"):
            children = getattr(obj, attr, None)
            if not children:
                continue
            if hasattr(children, "items"):
                iterable = children.items()
            else:
                iterable = enumerate(children)
            for name, child in iterable:
                yield from walk(f"{location}/{name}", child)

    for name, obj in nwbfile.acquisition.items():
        yield from walk(f"acquisition/{name}", obj)
    for module_name, module in nwbfile.processing.items():
        yield from walk(f"processing/{module_name}", module)


def meta_from_imaging_info(nwb_path: Path, info: NwbImagingSeriesInfo) -> VideoMeta:
    return VideoMeta(
        path=str(nwb_path),
        fps=float(info.fps),
        frame_count=int(info.n_frames),
        duration_ms=float(info.duration_ms),
        source_kind=SOURCE_NWB,
        series_location=str(info.location),
    )


def build_timestamp_mapping(
    imaging_start_s: float,
    ephys_start_s: float,
    signal_duration_ms: float,
    imaging_duration_ms: float,
) -> TimeMapping:
    """Identity clock map: shared session time via starting_time offsets.

    ``video_ms = signal_ms + (ephys_start - imaging_start) * 1000``.
    """
    b = (float(ephys_start_s) - float(imaging_start_s)) * 1000.0
    sig0 = max(0.0, -b)
    sig1 = min(float(signal_duration_ms), float(imaging_duration_ms) - b)
    if sig1 > sig0 + 1e-6:
        pairs = [(sig0, sig0 + b), (sig1, sig1 + b)]
    else:
        pairs = [(0.0, b)]
    return TimeMapping(pairs=pairs)


def ephys_location_from_header(header) -> str:
    name = str(getattr(header, "name_before_conversion", "") or "")
    if ":" not in name:
        return ""
    return name.split(":", 1)[1].strip()


def frame_to_uint8_bgr(frame: np.ndarray) -> np.ndarray:
    """Convert an imaging frame to contiguous uint8 BGR for Qt display."""
    arr = np.asarray(frame)
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    if arr.ndim == 3 and arr.shape[-1] >= 3:
        # Assume RGB-like; convert to BGR for consistency with OpenCV path
        rgb = arr[..., :3]
        if rgb.dtype != np.uint8:
            rgb = _scale_to_uint8(rgb)
        return np.ascontiguousarray(rgb[..., ::-1])
    if arr.ndim != 2:
        arr = np.squeeze(arr)
    if arr.ndim != 2:
        raise ValueError(f"Unsupported imaging frame shape: {np.asarray(frame).shape}")
    gray = _scale_to_uint8(arr)
    return np.stack([gray, gray, gray], axis=-1)


def _scale_to_uint8(arr: np.ndarray) -> np.ndarray:
    data = np.asarray(arr, dtype=np.float64)
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        return np.zeros(data.shape, dtype=np.uint8)
    lo = float(np.percentile(finite, 1.0))
    hi = float(np.percentile(finite, 99.0))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = float(np.min(finite))
        hi = float(np.max(finite))
    if hi <= lo:
        return np.zeros(data.shape, dtype=np.uint8)
    scaled = (np.clip(data, lo, hi) - lo) / (hi - lo)
    return np.ascontiguousarray((scaled * 255.0).astype(np.uint8))


class NwbFrameSource:
    """Keep an NWB file open and read imaging frames by index."""

    def __init__(self, nwb_path: Path, series_location: str):
        from pynwb import NWBHDF5IO

        self.path = Path(nwb_path)
        self.series_location = series_location
        self._io = NWBHDF5IO(str(self.path), "r", load_namespaces=True)
        nwbfile = self._io.read()
        self._series = _find_series(nwbfile, series_location)
        info = _series_info(series_location, self._series)
        if info is None:
            self.close()
            raise ValueError(f"Invalid imaging series: {series_location}")
        self.info = info

    def read_bgr(self, frame_index: int) -> Optional[np.ndarray]:
        idx = int(frame_index)
        if idx < 0 or idx >= self.info.n_frames:
            return None
        try:
            frame = np.asarray(self._series.data[idx])
            return frame_to_uint8_bgr(frame)
        except Exception:
            return None

    def close(self) -> None:
        try:
            if self._io is not None:
                self._io.close()
        finally:
            self._io = None
            self._series = None
