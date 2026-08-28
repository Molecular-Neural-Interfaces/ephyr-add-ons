"""Group-shared code for the Spike utils add-ons.

Holds the spike payload models, MAD-based detection algorithms and helpers to
locate/read the spike sets produced by the spike_detection and spike_importer
add-ons. Depends only on shared core infrastructure, never on another add-on
group.

Every spike set lives in its own directory under the experiment-wide
``add_ons/data/spike_sets`` folder, so that any spike add-on can read sets
regardless of which add-on produced them:

    add_ons/data/spike_sets/{set_name}/spike_set_meta.json
    add_ons/data/spike_sets/{set_name}/{sweep_idx}.spikes.json
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
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
SPIKE_SETS_FOLDER_NAME = "spike_sets"

SOURCE_DETECTED = "detected"
SOURCE_IMPORTED = "imported"

NO_SPIKE_SETS_MESSAGE = (
    "No spike sets yet. Run Spike detection, or import spikes with Spike importer first."
)


class SpikePoint(BaseModel):
    sample_idx: Optional[int] = None
    time_ms: float
    value: float
    polarity: str = "negative"
    # Sorting label inside the set: the detection method identifies the set,
    # while the cluster tells which unit the spike was assigned to (0 = unclustered).
    cluster: int = 0


class SpikesPayload(BaseModel):
    detector_name: str = "mad"
    source: str = SOURCE_DETECTED
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
    source_file: str = ""
    spikes_by_channel: Dict[int, List[SpikePoint]] = Field(default_factory=dict)

    def clusters(self) -> List[int]:
        found = {
            int(spike.cluster)
            for spikes in (self.spikes_by_channel or {}).values()
            for spike in spikes
        }
        return sorted(found)


SPIKE_SET_META_FILENAME = "spike_set_meta.json"


@dataclass
class SpikeSetMeta:
    """Description of one spike set, shared by every spike add-on."""

    source: str = SOURCE_DETECTED
    detector_name: str = "mad"
    group_key: str = ""
    group_name: str = ""
    preprocessing_pipeline: str = DEFAULT_PIPELINE_NAME
    threshold: float = 6.0
    adaptive_sigma: bool = True
    clusters: List[int] = field(default_factory=list)
    source_file: str = ""
    created_at: str = ""

    @property
    def is_imported(self) -> bool:
        return str(self.source) == SOURCE_IMPORTED

    def display_label(self, fallback_name: str = "") -> str:
        if self.is_imported:
            parts = [(self.detector_name or "imported").strip(), SOURCE_IMPORTED]
            if self.source_file:
                parts.append(Path(self.source_file).name)
            parts.append(self._clusters_label())
            return " | ".join(part for part in parts if part)

        group = (self.group_name or "").strip() or "Group"
        pipeline = (self.preprocessing_pipeline or "").strip() or DEFAULT_PIPELINE_NAME
        mode = "adaptive" if self.adaptive_sigma else "global"
        mult = f"{float(self.threshold):.6f}".rstrip("0").rstrip(".")
        parts = [group, SOURCE_DETECTED, pipeline, f"{mode} MAD×{mult}", self._clusters_label()]
        label = " | ".join(part for part in parts if part)
        return label if (self.group_name or self.preprocessing_pipeline) else (fallback_name or label)

    def _clusters_label(self) -> str:
        real_clusters = [cluster for cluster in (self.clusters or []) if int(cluster) != 0]
        if not real_clusters:
            return ""
        return f"{len(real_clusters)} clusters" if len(real_clusters) > 1 else "1 cluster"


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
    # Interp in chunks to avoid allocating a full np.arange(n) float64 buffer.
    sigma_full = np.empty(n, dtype=np.float64)
    centers_arr = np.asarray(centers, dtype=np.float64)
    chunk = max(65_536, n // 8 if n > 0 else 65_536)
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        sigma_full[start:stop] = np.interp(
            np.arange(start, stop, dtype=np.float64),
            centers_arr,
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


def _sanitize_name_part(value: str, fallback: str) -> str:
    clean = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in (value or "").strip())
    return clean or fallback


def safe_detected_set_dir_name(
    pipeline_name: str,
    threshold: float,
    adaptive: bool,
    group_name: str = "",
) -> str:
    clean_group = _sanitize_name_part(group_name, "group")
    clean_pipe = _sanitize_name_part(pipeline_name, "raw")
    mult = f"{float(threshold):.6f}".rstrip("0").rstrip(".").replace(".", "_")
    mode = "adaptive" if adaptive else "global"
    return f"{clean_group}__{clean_pipe}_{mode}_mad_{mult}"


def safe_imported_set_dir_name(source_file_name: str, method_name: str) -> str:
    clean_source = _sanitize_name_part(Path(source_file_name).stem, "spikes")
    clean_method = _sanitize_name_part(method_name, "imported")
    return f"{SOURCE_IMPORTED}__{clean_source}__{clean_method}"


# ---- Base with spike-set locating/reading ----

class SpikeUtilsBase(EphyrAddOnMixin):
    def spike_sets_dir(self, add_on_data_dir: Path) -> Path:
        """Experiment-wide folder holding every spike set, whatever produced it."""
        name = Path(add_on_data_dir).name
        prefix = "dev_" if name.startswith("dev_") else ""
        return Path(add_on_data_dir).parent / f"{prefix}{SPIKE_SETS_FOLDER_NAME}"

    def list_spike_set_dirs(self, add_on_data_dir: Path) -> List[Path]:
        base = self.spike_sets_dir(add_on_data_dir)
        if not base.exists():
            return []
        return sorted([p for p in base.iterdir() if p.is_dir()], key=lambda p: p.name)

    def spike_set_meta_path(self, set_dir: Path) -> Path:
        return Path(set_dir) / SPIKE_SET_META_FILENAME

    def save_spike_set_meta(self, set_dir: Path, meta: SpikeSetMeta) -> None:
        path = self.spike_set_meta_path(set_dir)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "source": str(meta.source or SOURCE_DETECTED),
            "detector_name": str(meta.detector_name or "mad"),
            "group_key": str(meta.group_key or ""),
            "group_name": str(meta.group_name or ""),
            "preprocessing_pipeline": str(meta.preprocessing_pipeline or DEFAULT_PIPELINE_NAME),
            "threshold": float(meta.threshold),
            "adaptive_sigma": bool(meta.adaptive_sigma),
            "clusters": [int(cluster) for cluster in (meta.clusters or [])],
            "source_file": str(meta.source_file or ""),
            "created_at": str(meta.created_at or ""),
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def read_spike_set_meta(self, set_dir: Path) -> SpikeSetMeta:
        path = self.spike_set_meta_path(set_dir)
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                return SpikeSetMeta(
                    source=str(data.get("source", SOURCE_DETECTED) or SOURCE_DETECTED),
                    detector_name=str(data.get("detector_name", "mad") or "mad"),
                    group_key=str(data.get("group_key", "") or ""),
                    group_name=str(data.get("group_name", "") or ""),
                    preprocessing_pipeline=str(
                        data.get("preprocessing_pipeline", DEFAULT_PIPELINE_NAME) or DEFAULT_PIPELINE_NAME
                    ),
                    threshold=float(data.get("threshold", 6.0)),
                    adaptive_sigma=bool(data.get("adaptive_sigma", True)),
                    clusters=[int(cluster) for cluster in (data.get("clusters") or [])],
                    source_file=str(data.get("source_file", "") or ""),
                    created_at=str(data.get("created_at", "") or ""),
                )
            except Exception as e:
                ephyr_logger().debug(str(e))

        # Fallback: infer from any sweep payload in the directory.
        for spikes_path in sorted(Path(set_dir).glob("*.spikes.json")):
            try:
                payload = SpikesPayload.model_validate_json(spikes_path.read_text(encoding="utf-8"))
                return SpikeSetMeta(
                    source=str(payload.source or SOURCE_DETECTED),
                    detector_name=str(payload.detector_name or "mad"),
                    group_key=str(payload.group_key or ""),
                    group_name=str(payload.group_name or ""),
                    preprocessing_pipeline=str(payload.preprocessing_pipeline or DEFAULT_PIPELINE_NAME),
                    threshold=float(payload.threshold),
                    adaptive_sigma=bool(payload.adaptive_sigma),
                    clusters=payload.clusters(),
                    source_file=str(payload.source_file or ""),
                )
            except Exception as e:
                ephyr_logger().debug(str(e))
        return SpikeSetMeta()

    def spike_set_label(self, set_dir: Path) -> str:
        meta = self.read_spike_set_meta(set_dir)
        return meta.display_label(fallback_name=Path(set_dir).name)

    def read_spikes_payload(self, set_dir: Path, sweep_idx: int) -> Optional[SpikesPayload]:
        path = Path(set_dir) / f"{int(sweep_idx)}.spikes.json"
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

    def choose_spike_set_dialog(
        self,
        title: str,
        add_on_data_dir: Path,
        selected_dir: str = "",
        label: str = "Spike set:",
    ) -> Optional[Path]:
        dirs = self.list_spike_set_dirs(add_on_data_dir)
        if not dirs:
            QMessageBox.warning(None, title, NO_SPIKE_SETS_MESSAGE)
            return None
        dialog = QDialog()
        dialog.setWindowTitle(title)
        layout = QVBoxLayout(dialog)
        form = QFormLayout()
        combo = QComboBox()
        for path in dirs:
            combo.addItem(self.spike_set_label(path), str(path))
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

    def choose_spike_sets_dialog(
        self,
        title: str,
        add_on_data_dir: Path,
        selected_dirs: Optional[List[str]] = None,
        label: str = "Spike sets:",
    ) -> Optional[List[Path]]:
        dirs = self.list_spike_set_dirs(add_on_data_dir)
        if not dirs:
            QMessageBox.warning(None, title, NO_SPIKE_SETS_MESSAGE)
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
            item = QListWidgetItem(self.spike_set_label(path))
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
            QMessageBox.warning(dialog, title, "Select at least one spike set.")
            return None
        return selected

    def channels_for_spikes_payload(
        self,
        payload: SpikesPayload,
        channel_groups: Optional[List[Any]] = None,
    ) -> List[int]:
        """Channels belonging to the set's channel group (fallback: payload keys)."""
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
    "NO_SPIKE_SETS_MESSAGE",
    "SOURCE_DETECTED",
    "SOURCE_IMPORTED",
    "SPIKE_SETS_FOLDER_NAME",
    "SPIKE_SET_META_FILENAME",
    "SpikeCandidate",
    "SpikePoint",
    "SpikeSetMeta",
    "SpikesPayload",
    "SpikeUtilsBase",
    "detect_spikes_adaptive_mad",
    "detect_spikes_mad",
    "mad_sigma",
    "merge_spikes_global",
    "rolling_sigma_mad",
    "safe_detected_set_dir_name",
    "safe_imported_set_dir_name",
]
