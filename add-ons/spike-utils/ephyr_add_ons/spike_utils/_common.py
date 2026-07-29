"""Group-shared code for the Spike utils add-ons.

Holds the spike payload models, MAD-based detection algorithms and helpers to
locate/read detection results produced by the spike_detection add-on. Depends
only on shared core infrastructure, never on another add-on group.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from pydantic import BaseModel, Field
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QFormLayout,
    QHBoxLayout,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
)
from scipy.signal import find_peaks

from ephyr.core.add_ons import EphyrAddOnMixin
from ephyr.logger import ephyr_logger

DEFAULT_PIPELINE_NAME = "raw"
DETECTION_MODULE_NAME = "spike_detection"


class SpikePoint(BaseModel):
    sample_idx: Optional[int] = None
    time_ms: float
    value: float
    polarity: str = "negative"


class SpikesPayload(BaseModel):
    detector_name: str = "mad"
    preprocessing_pipeline: str = DEFAULT_PIPELINE_NAME
    threshold: float = 6.0
    sweep_idx: int
    sample_rate: float
    adaptive_sigma: bool = True
    sigma_params: Dict[str, Any] = Field(default_factory=dict)
    detect_positive: bool = False
    detect_negative: bool = True
    merge_window_ms: float = 1.0
    group_key: str = ""
    group_name: str = ""
    ignore_event_names: List[str] = Field(default_factory=list)
    ignore_before_ms: float = 0.0
    ignore_after_ms: float = 0.0
    ignore_period_names: List[str] = Field(default_factory=list)
    events_mode: str = "ignore"
    periods_mode: str = "ignore"
    spikes_by_channel: Dict[int, List[SpikePoint]] = Field(default_factory=dict)


DETECTION_META_FILENAME = "detection_meta.json"


@dataclass
class DetectionResultMeta:
    group_key: str = ""
    group_name: str = ""
    preprocessing_pipeline: str = DEFAULT_PIPELINE_NAME
    threshold: float = 6.0
    adaptive_sigma: bool = True
    detector_name: str = "mad"

    def display_label(self, fallback_name: str = "") -> str:
        group = (self.group_name or "").strip() or "Group"
        pipeline = (self.preprocessing_pipeline or "").strip() or DEFAULT_PIPELINE_NAME
        mode = "adaptive" if self.adaptive_sigma else "global"
        mult = f"{float(self.threshold):.6f}".rstrip("0").rstrip(".")
        label = f"{group} | {pipeline} | {mode} MAD×{mult}"
        return label if (self.group_name or self.preprocessing_pipeline) else (fallback_name or label)


@dataclass
class SpikeCandidate:
    sample_idx: int
    value: float
    polarity: str = "negative"


# ---- Detection algorithms ----

def mad_sigma(signal_1d: np.ndarray, sigma_floor_uv: float = 0.0) -> float:
    x = np.asarray(signal_1d, dtype=np.float64)
    if x.size == 0:
        return float(sigma_floor_uv)
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    return max(float(sigma_floor_uv), float(mad / 0.6745 if mad > 0 else 0.0))


def rolling_sigma_mad(
    signal_1d: np.ndarray,
    fs: float,
    window_ms: float = 500.0,
    step_ms: float = 100.0,
    sigma_floor_uv: float = 2.0,
    smooth_windows: int = 3,
    mask: Optional[np.ndarray] = None,
    min_valid_fraction: float = 0.20,
) -> np.ndarray:
    x = np.asarray(signal_1d, dtype=np.float64)
    n = x.size
    if n == 0:
        return np.array([], dtype=np.float64)
    valid_mask = None
    if mask is not None:
        valid_mask = np.asarray(mask, dtype=bool)
        if valid_mask.size != n:
            valid_mask = None
    w = min(n, max(8, int(round(float(window_ms) * float(fs) / 1000.0))))
    h = max(1, int(round(float(step_ms) * float(fs) / 1000.0)))
    centers: List[int] = []
    sigma_vals: List[float] = []
    prev_sigma = float(sigma_floor_uv)
    min_valid_n = max(8, int(round(float(min_valid_fraction) * w)))
    for start in range(0, max(1, n - w + 1), h):
        seg = x[start:start + w]
        seg_valid = seg[valid_mask[start:start + w]] if valid_mask is not None else seg
        if seg_valid.size >= min_valid_n:
            med = np.median(seg_valid)
            mad = np.median(np.abs(seg_valid - med))
            sigma = mad / 0.6745 if mad > 0 else 0.0
            prev_sigma = max(float(sigma), float(sigma_floor_uv))
        centers.append(start + w // 2)
        sigma_vals.append(prev_sigma)
    if not centers:
        return np.full(n, float(sigma_floor_uv), dtype=np.float64)
    sigma_arr = np.asarray(sigma_vals, dtype=np.float64)
    if int(smooth_windows) > 1 and sigma_arr.size > 1:
        k = int(max(1, smooth_windows))
        sigma_arr = np.convolve(sigma_arr, np.ones(k) / float(k), mode="same")
    sigma_full = np.interp(
        np.arange(n, dtype=np.float64),
        np.asarray(centers, dtype=np.float64),
        sigma_arr,
        left=sigma_arr[0],
        right=sigma_arr[-1],
    )
    global_sigma = max(float(sigma_floor_uv), float(np.median(sigma_arr)))
    lock_n = max(0, min(int(round(w / 2)), n))
    if lock_n > 0:
        sigma_full[:lock_n] = np.maximum(sigma_full[:lock_n], global_sigma)
        sigma_full[n - lock_n:] = np.maximum(sigma_full[n - lock_n:], global_sigma)
    return sigma_full


def _candidates_from_indices(signal_1d: np.ndarray, indices: Sequence[int], polarity: str) -> List[SpikeCandidate]:
    x = np.asarray(signal_1d, dtype=np.float64)
    out: List[SpikeCandidate] = []
    for idx in np.asarray(indices, dtype=np.int64):
        if 0 <= int(idx) < x.size:
            out.append(SpikeCandidate(sample_idx=int(idx), value=float(x[int(idx)]), polarity=polarity))
    return out


def detect_spikes_mad(
    signal_1d: np.ndarray,
    fs: float,
    multiplier: float = 6.0,
    min_distance_ms: float = 1.0,
    detect_positive: bool = False,
    sigma_floor_uv: float = 0.0,
) -> List[SpikeCandidate]:
    x = np.asarray(signal_1d, dtype=np.float64)
    sigma = mad_sigma(x, sigma_floor_uv=sigma_floor_uv)
    threshold = float(multiplier) * sigma
    distance = max(1, int(round(float(min_distance_ms) * float(fs) / 1000.0)))
    if detect_positive:
        peaks, _props = find_peaks(x, height=threshold, distance=distance)
        return _candidates_from_indices(x, peaks, "positive")
    peaks, _props = find_peaks(-x, height=threshold, distance=distance)
    return _candidates_from_indices(x, peaks, "negative")


def detect_spikes_adaptive_mad(
    signal_1d: np.ndarray,
    fs: float,
    multiplier: float = 6.0,
    min_distance_ms: float = 1.0,
    detect_positive: bool = False,
    sigma_t: Optional[np.ndarray] = None,
    sigma_floor_uv: float = 2.0,
) -> List[SpikeCandidate]:
    x = np.asarray(signal_1d, dtype=np.float64)
    if x.size == 0:
        return []
    sigma_arr = (
        rolling_sigma_mad(x, fs, sigma_floor_uv=sigma_floor_uv)
        if sigma_t is None
        else np.asarray(sigma_t, dtype=np.float64)
    )
    if sigma_arr.size != x.size:
        sigma_arr = np.full(x.size, mad_sigma(x, sigma_floor_uv=sigma_floor_uv), dtype=np.float64)
    thr = float(multiplier) * sigma_arr
    distance = max(1, int(round(float(min_distance_ms) * float(fs) / 1000.0)))
    mask = x > thr if detect_positive else (-x > thr)
    candidate = np.flatnonzero(mask)
    if candidate.size == 0:
        return []
    picked: List[int] = []
    best_idx = int(candidate[0])
    best_score = float((x[best_idx] - thr[best_idx]) if detect_positive else (-x[best_idx] - thr[best_idx]))
    for raw_idx in candidate[1:]:
        idx = int(raw_idx)
        score = float((x[idx] - thr[idx]) if detect_positive else (-x[idx] - thr[idx]))
        if idx - best_idx <= distance:
            if score > best_score:
                best_idx, best_score = idx, score
        else:
            picked.append(best_idx)
            best_idx, best_score = idx, score
    picked.append(best_idx)
    return _candidates_from_indices(x, picked, "positive" if detect_positive else "negative")


def merge_spikes_global(spikes: Sequence[SpikeCandidate], fs: float, min_distance_ms: float = 1.0) -> List[SpikeCandidate]:
    rows = sorted(list(spikes), key=lambda sp: int(sp.sample_idx))
    if not rows:
        return []
    distance = max(1, int(round(float(min_distance_ms) * float(fs) / 1000.0)))
    picked: List[SpikeCandidate] = []
    best = rows[0]
    for sp in rows[1:]:
        if int(sp.sample_idx) - int(best.sample_idx) <= distance:
            if abs(float(sp.value)) > abs(float(best.value)):
                best = sp
        else:
            picked.append(best)
            best = sp
    picked.append(best)
    return picked


def safe_detection_dir_name(
    pipeline_name: str,
    threshold: float,
    adaptive: bool,
    group_name: str = "",
) -> str:
    def _sanitize(value: str, fallback: str) -> str:
        clean = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in (value or "").strip())
        return clean or fallback

    clean_group = _sanitize(group_name, "group")
    clean_pipe = _sanitize(pipeline_name, "raw")
    mult = f"{float(threshold):.6f}".rstrip("0").rstrip(".").replace(".", "_")
    mode = "adaptive" if adaptive else "global"
    return f"{clean_group}__{clean_pipe}_{mode}_mad_{mult}"


# ---- Base with detection-result locating/reading ----

class SpikeUtilsBase(EphyrAddOnMixin):
    def detection_dir(self, add_on_data_dir: Path) -> Path:
        name = Path(add_on_data_dir).name
        prefix = "dev_" if name.startswith("dev_") else ""
        return Path(add_on_data_dir).parent / f"{prefix}{DETECTION_MODULE_NAME}"

    def list_detection_result_dirs(self, add_on_data_dir: Path) -> List[Path]:
        base = self.detection_dir(add_on_data_dir)
        if not base.exists():
            return []
        return sorted([p for p in base.iterdir() if p.is_dir()], key=lambda p: p.name)

    def detection_meta_path(self, result_dir: Path) -> Path:
        return Path(result_dir) / DETECTION_META_FILENAME

    def save_detection_meta(self, result_dir: Path, meta: DetectionResultMeta) -> None:
        path = self.detection_meta_path(result_dir)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "group_key": str(meta.group_key or ""),
            "group_name": str(meta.group_name or ""),
            "preprocessing_pipeline": str(meta.preprocessing_pipeline or DEFAULT_PIPELINE_NAME),
            "threshold": float(meta.threshold),
            "adaptive_sigma": bool(meta.adaptive_sigma),
            "detector_name": str(meta.detector_name or "mad"),
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def read_detection_meta(self, result_dir: Path) -> DetectionResultMeta:
        path = self.detection_meta_path(result_dir)
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                return DetectionResultMeta(
                    group_key=str(data.get("group_key", "") or ""),
                    group_name=str(data.get("group_name", "") or ""),
                    preprocessing_pipeline=str(
                        data.get("preprocessing_pipeline", DEFAULT_PIPELINE_NAME) or DEFAULT_PIPELINE_NAME
                    ),
                    threshold=float(data.get("threshold", 6.0)),
                    adaptive_sigma=bool(data.get("adaptive_sigma", True)),
                    detector_name=str(data.get("detector_name", "mad") or "mad"),
                )
            except Exception as e:
                ephyr_logger().debug(str(e))

        # Fallback: infer from any sweep payload in the directory.
        for spikes_path in sorted(Path(result_dir).glob("*.spikes.json")):
            try:
                payload = SpikesPayload.model_validate_json(spikes_path.read_text(encoding="utf-8"))
                return DetectionResultMeta(
                    group_key=str(payload.group_key or ""),
                    group_name=str(payload.group_name or ""),
                    preprocessing_pipeline=str(payload.preprocessing_pipeline or DEFAULT_PIPELINE_NAME),
                    threshold=float(payload.threshold),
                    adaptive_sigma=bool(payload.adaptive_sigma),
                    detector_name=str(payload.detector_name or "mad"),
                )
            except Exception as e:
                ephyr_logger().debug(str(e))
        return DetectionResultMeta()

    def detection_result_label(self, result_dir: Path) -> str:
        meta = self.read_detection_meta(result_dir)
        return meta.display_label(fallback_name=Path(result_dir).name)

    def read_spikes_payload(self, result_dir: Path, sweep_idx: int) -> Optional[SpikesPayload]:
        path = Path(result_dir) / f"{int(sweep_idx)}.spikes.json"
        if not path.exists():
            return None
        try:
            return SpikesPayload.model_validate_json(path.read_text(encoding="utf-8"))
        except Exception as e:
            ephyr_logger().debug(str(e))
            return None

    def save_spikes_payload(self, path: Path, payload: SpikesPayload) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(payload.model_dump_json(indent=2), encoding="utf-8")

    def choose_result_dir_dialog(
        self,
        title: str,
        add_on_data_dir: Path,
        selected_dir: str = "",
        label: str = "Detection method:",
    ) -> Optional[Path]:
        dirs = self.list_detection_result_dirs(add_on_data_dir)
        if not dirs:
            QMessageBox.warning(None, title, "No detected spikes yet. Run Spike detection first.")
            return None
        dialog = QDialog()
        dialog.setWindowTitle(title)
        layout = QVBoxLayout(dialog)
        form = QFormLayout()
        combo = QComboBox()
        for path in dirs:
            combo.addItem(self.detection_result_label(path), str(path))
        if selected_dir:
            idx = combo.findData(selected_dir)
            combo.setCurrentIndex(max(0, idx))
        form.addRow(label, combo)
        layout.addLayout(form)
        actions = QHBoxLayout()
        btn_cancel = QPushButton("Cancel")
        btn_ok = QPushButton("Select")
        actions.addStretch(1)
        actions.addWidget(btn_cancel)
        actions.addWidget(btn_ok)
        layout.addLayout(actions)
        btn_cancel.clicked.connect(dialog.reject)
        btn_ok.clicked.connect(dialog.accept)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return None
        return Path(str(combo.currentData()))

    def choose_result_dirs_dialog(
        self,
        title: str,
        add_on_data_dir: Path,
        selected_dirs: Optional[List[str]] = None,
        label: str = "Detection methods:",
    ) -> Optional[List[Path]]:
        dirs = self.list_detection_result_dirs(add_on_data_dir)
        if not dirs:
            QMessageBox.warning(None, title, "No detected spikes yet. Run Spike detection first.")
            return None
        preferred = {str(Path(p)) for p in (selected_dirs or []) if p}
        dialog = QDialog()
        dialog.setWindowTitle(title)
        layout = QVBoxLayout(dialog)
        form = QFormLayout()
        methods_list = QListWidget()
        methods_list.setSelectionMode(QListWidget.SelectionMode.MultiSelection)
        select_all = not preferred
        for path in dirs:
            item = QListWidgetItem(self.detection_result_label(path))
            item.setData(Qt.ItemDataRole.UserRole, str(path))
            item.setSelected(select_all or str(path) in preferred)
            methods_list.addItem(item)
        form.addRow(label, methods_list)
        layout.addLayout(form)
        actions = QHBoxLayout()
        btn_cancel = QPushButton("Cancel")
        btn_ok = QPushButton("Select")
        actions.addStretch(1)
        actions.addWidget(btn_cancel)
        actions.addWidget(btn_ok)
        layout.addLayout(actions)
        btn_cancel.clicked.connect(dialog.reject)
        btn_ok.clicked.connect(dialog.accept)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return None
        selected: List[Path] = []
        for row in range(methods_list.count()):
            item = methods_list.item(row)
            if item is None or not item.isSelected():
                continue
            value = item.data(Qt.ItemDataRole.UserRole)
            if value:
                selected.append(Path(str(value)))
        if not selected:
            QMessageBox.warning(dialog, title, "Select at least one detection method.")
            return None
        return selected

    def channels_for_detection_payload(
        self,
        payload: SpikesPayload,
        channel_groups: Optional[List[Any]] = None,
    ) -> List[int]:
        """Channels belonging to the detection group (fallback: payload keys)."""
        if channel_groups is not None and payload.group_key:
            resolved = self.resolve_groups_by_keys(
                channel_groups,
                [payload.group_key],
                non_aux_only=False,
                fallback_all=False,
            )
            if resolved:
                return [int(ch) for ch in (getattr(resolved[0], "channel_indexes", []) or [])]
        return sorted(int(ch) for ch in (payload.spikes_by_channel or {}).keys())


__all__ = [
    "DEFAULT_PIPELINE_NAME",
    "DETECTION_META_FILENAME",
    "DETECTION_MODULE_NAME",
    "DetectionResultMeta",
    "SpikeCandidate",
    "SpikePoint",
    "SpikesPayload",
    "SpikeUtilsBase",
    "detect_spikes_adaptive_mad",
    "detect_spikes_mad",
    "mad_sigma",
    "merge_spikes_global",
    "rolling_sigma_mad",
    "safe_detection_dir_name",
]
