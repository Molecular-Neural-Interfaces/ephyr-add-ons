"""Events detection add-on.

Runnable. Two detection modes:
- TTL: threshold crossings on rising or falling edge
- Above threshold: peak detection above a height with min distance

Detected events are appended to the session vocabulary; if more than 1000
events are found, the user must confirm before adding them.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)
from scipy.signal import find_peaks

from ephyr.core.add_ons.base import BaseAddOn
from ephyr.core.add_ons import (
    PipelineSelector,
    apply_preprocessing_pipeline,
    read_pipeline_store,
)
from ephyr.logger import ephyr_logger

from ephyr_add_ons.labeling_utils._common import LabelingUtilsBase

CONFIRM_THRESHOLD = 1000
MODE_TTL = "ttl"
MODE_ABOVE = "above_threshold"
EDGE_RISING = "rising"
EDGE_FALLING = "falling"


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


def detect_above_threshold(
    signal_1d: np.ndarray,
    height: float,
    min_distance_samples: int = 1,
) -> np.ndarray:
    """Return peak indices above ``height`` (supports negative height)."""
    x = np.asarray(signal_1d, dtype=np.float64)
    search = x
    search_height = float(height)
    if height < 0:
        search = -x
        search_height = -height
    peaks, _props = find_peaks(
        search,
        height=search_height,
        distance=max(1, int(min_distance_samples)),
    )
    return np.asarray(peaks, dtype=np.int64)


class _EventsDetectionWindow(QWidget):
    def __init__(self, add_on, session_manager, header, add_on_data_dir: Path, groups):
        super().__init__()
        self._add_on = add_on
        self._session_manager = session_manager
        self._header = header
        self._add_on_data_dir = Path(add_on_data_dir)

        common = add_on.load_common(self._add_on_data_dir)
        params = add_on.load_params(self._add_on_data_dir)

        self.setWindowTitle("Events detection")
        self.setWindowFlags(Qt.WindowType.Window)
        self.setMinimumWidth(540)
        self.setMinimumHeight(620)
        outer = QVBoxLayout(self)
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_widget = QWidget()
        form = QFormLayout(scroll_widget)

        self._group_combo, self._channels_list = add_on.build_group_channel_selector(
            form,
            groups,
            header,
            preferred_group_idx=int(common.get("group_idx", 0)),
            preferred_channels=common.get("channel_indexes", []),
        )

        self._pipeline_selector = PipelineSelector(
            add_on.pipelines_path(self._add_on_data_dir),
            selected_name=str(params.get("pipeline", "raw")),
        )
        form.addRow("Preprocessing pipeline:", self._pipeline_selector)

        self._mode_combo = QComboBox()
        self._mode_combo.addItem("TTL", MODE_TTL)
        self._mode_combo.addItem("Above threshold", MODE_ABOVE)
        saved_mode = str(params.get("mode", MODE_TTL))
        if saved_mode in {"digital", "analog"}:
            saved_mode = MODE_ABOVE if saved_mode == "digital" else MODE_TTL
        idx = self._mode_combo.findData(saved_mode)
        self._mode_combo.setCurrentIndex(max(0, idx))
        form.addRow("Detection mode:", self._mode_combo)

        self._name_edit = QLineEdit(str(params.get("event_name", "")))
        form.addRow("Event name:", self._name_edit)

        self._edge_combo = QComboBox()
        self._edge_combo.addItem("Rising edge", EDGE_RISING)
        self._edge_combo.addItem("Falling edge", EDGE_FALLING)
        edge_idx = self._edge_combo.findData(str(params.get("edge", EDGE_RISING)))
        self._edge_combo.setCurrentIndex(max(0, edge_idx))
        form.addRow("TTL edge:", self._edge_combo)

        self._ttl_threshold_spin = QDoubleSpinBox()
        self._ttl_threshold_spin.setDecimals(6)
        self._ttl_threshold_spin.setRange(-1e9, 1e9)
        self._ttl_threshold_spin.setValue(float(params.get("ttl_threshold", params.get("height", 0.5))))
        form.addRow("TTL threshold:", self._ttl_threshold_spin)

        self._height_spin = QDoubleSpinBox()
        self._height_spin.setDecimals(6)
        self._height_spin.setRange(-1e9, 1e9)
        self._height_spin.setValue(float(params.get("height", 50.0)))
        form.addRow("Height (threshold):", self._height_spin)

        self._distance_spin = QDoubleSpinBox()
        self._distance_spin.setDecimals(3)
        self._distance_spin.setRange(0.0, 1e9)
        self._distance_spin.setValue(float(params.get("distance_ms", 1.0)))
        form.addRow("Min distance (ms):", self._distance_spin)

        self._mode_combo.currentIndexChanged.connect(lambda _i: self._sync_mode_visibility())
        self._sync_mode_visibility()

        scroll_area.setWidget(scroll_widget)
        outer.addWidget(scroll_area)

        self._status = QLabel("")
        self._status.setWordWrap(True)
        outer.addWidget(self._status)
        self._progress = QProgressBar()
        self._progress.setRange(0, 100)
        self._progress.setValue(0)
        outer.addWidget(self._progress)

        actions = QHBoxLayout()
        btn_close = QPushButton("Close")
        self._run_btn = QPushButton("Run")
        actions.addStretch(1)
        actions.addWidget(btn_close)
        actions.addWidget(self._run_btn)
        outer.addLayout(actions)
        btn_close.clicked.connect(self.close)
        self._run_btn.clicked.connect(self._on_run)

    def _sync_mode_visibility(self) -> None:
        is_ttl = str(self._mode_combo.currentData()) == MODE_TTL
        self._edge_combo.setEnabled(is_ttl)
        self._ttl_threshold_spin.setEnabled(is_ttl)
        self._height_spin.setEnabled(not is_ttl)

    def _set_status(self, progress: Optional[int], message: str) -> None:
        self._status.setText(message)
        if progress is not None:
            self._progress.setValue(max(0, min(100, int(progress))))
        QApplication.processEvents()

    def _collect_params(self) -> Optional[dict]:
        channels = self._add_on.selected_channels(self._channels_list)
        if not channels:
            QMessageBox.warning(self, "Events detection", "Select at least one channel.")
            return None
        event_name = self._name_edit.text().strip()
        if not event_name:
            QMessageBox.warning(self, "Events detection", "Event name must not be empty.")
            return None
        existing = {entry.name for entry in (self._session_manager.events_vocabulary or {}).values()}
        if event_name in existing:
            QMessageBox.warning(self, "Events detection", "Event name must be unique.")
            return None
        self._add_on.save_common(
            self._add_on_data_dir,
            {"group_idx": int(self._group_combo.currentData()), "channel_indexes": channels},
        )
        payload = {
            "pipeline": self._pipeline_selector.current_pipeline_name(),
            "mode": str(self._mode_combo.currentData()),
            "event_name": event_name,
            "edge": str(self._edge_combo.currentData()),
            "ttl_threshold": float(self._ttl_threshold_spin.value()),
            "height": float(self._height_spin.value()),
            "distance_ms": float(self._distance_spin.value()),
        }
        self._add_on.save_params(self._add_on_data_dir, payload)
        return {"channels": channels, **payload}

    def _on_run(self) -> None:
        params = self._collect_params()
        if params is None:
            return
        self._run_btn.setEnabled(False)
        try:
            for yielded in self._add_on.detect_events(
                self._session_manager, self._add_on_data_dir, params, parent=self
            ):
                if isinstance(yielded, dict):
                    self._set_status(yielded.get("progress"), str(yielded.get("message", "")))
        finally:
            self._run_btn.setEnabled(True)


class EventsDetectionAddOn(LabelingUtilsBase, BaseAddOn):
    TRANSFORMATION = False
    VIEWABLE = False
    RUNNABLE = True

    def __init__(self):
        self._window: Optional[_EventsDetectionWindow] = None

    def detect_events(self, session_manager, add_on_data_dir: Path, params: dict, parent=None):
        header = session_manager.header
        add_on_data_dir = Path(add_on_data_dir)
        total_sweeps = int(header.number_of_sweeps)
        if total_sweeps <= 0:
            yield {"progress": 100, "message": "No sweeps to process"}
            return

        sample_rate = float(header.sample_rate)
        channels = params["channels"]
        pipeline = read_pipeline_store(self.pipelines_path(add_on_data_dir)).get(params["pipeline"])
        mode = str(params["mode"])
        distance_samples = (
            max(1, int((params["distance_ms"] * sample_rate) / 1000.0))
            if params["distance_ms"] > 0
            else 1
        )

        detected: List[Tuple[int, float]] = []
        for sweep_idx in range(total_sweeps):
            sweep_points = int(header.number_of_points_per_sweep[sweep_idx])
            yield {
                "progress": int((sweep_idx / total_sweeps) * 90),
                "message": f"Preprocessing pipeline '{params['pipeline']}' on sweep {sweep_idx + 1}/{total_sweeps}",
            }
            matrix = self.channel_matrix_from_session(
                session_manager, channels, sweep_idx, 0, sweep_points, sample_rate
            )
            processed = apply_preprocessing_pipeline(matrix, sample_rate, pipeline)
            processed = np.asarray(processed, dtype=np.float64)

            for row_idx in range(processed.shape[0]):
                signal = processed[row_idx]
                try:
                    if mode == MODE_TTL:
                        peaks = detect_ttl_edges(
                            signal,
                            threshold=float(params["ttl_threshold"]),
                            edge=str(params["edge"]),
                            min_distance_samples=distance_samples,
                        )
                    else:
                        peaks = detect_above_threshold(
                            signal,
                            height=float(params["height"]),
                            min_distance_samples=distance_samples,
                        )
                except Exception as e:
                    ephyr_logger().debug(str(e))
                    continue
                for peak_sample in peaks:
                    time_ms = (float(peak_sample) * 1000.0) / sample_rate
                    detected.append((sweep_idx, float(time_ms)))
            yield {
                "progress": int(((sweep_idx + 1) / total_sweeps) * 90),
                "message": f"Sweep {sweep_idx + 1}/{total_sweeps}: {len(detected)} events so far",
            }

        if not detected:
            QMessageBox.information(
                parent,
                "Events detection",
                "No events were detected. Vocabulary was not created.",
            )
            return

        n_detected = len(detected)
        if n_detected > CONFIRM_THRESHOLD:
            answer = QMessageBox.question(
                parent,
                "Events detection",
                f"Detected {n_detected} events. The interface may freeze. "
                "Are you sure you want to continue with these detection parameters?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if answer != QMessageBox.StandardButton.Yes:
                yield {"progress": 100, "message": f"Cancelled: {n_detected} events not added"}
                return

        yield {"progress": 95, "message": f"Adding {n_detected} events..."}
        try:
            event_name_id = session_manager.add_event_vocabulary(params["event_name"])
            events_specs = [(event_name_id, sweep_idx, time_ms) for sweep_idx, time_ms in detected]
            session_manager.add_events(events_specs)
        except Exception as e:
            ephyr_logger().debug(str(e))
            yield {"progress": 100, "message": "Failed to add events"}
            return
        yield {"progress": 100, "message": f"Added {n_detected} '{params['event_name']}' events"}

    def run(self, session_manager, add_on_data_dir):
        header = session_manager.header
        add_on_data_dir = Path(add_on_data_dir)
        groups = self.ensure_non_aux_groups(session_manager, "Events detection")
        if groups is None:
            return False
        try:
            if self._window is not None:
                self._window.close()
        except Exception as e:
            ephyr_logger().debug(str(e))
        self._window = _EventsDetectionWindow(self, session_manager, header, add_on_data_dir, groups)
        self._window.show()
        self._window.raise_()
        self._window.activateWindow()
        return False
