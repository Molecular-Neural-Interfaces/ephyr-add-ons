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
from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLineEdit,
    QMessageBox,
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


class EventsDetectionAddOn(LabelingUtilsBase, BaseAddOn):
    TRANSFORMATION = False
    VIEWABLE = False
    RUNNABLE = True

    def _ask_parameters(self, session_manager, header, add_on_data_dir: Path) -> Optional[dict]:
        groups = self.ensure_non_aux_groups(session_manager, "Events detection")
        if groups is None:
            return None
        common = self.load_common(add_on_data_dir)
        params = self.load_params(add_on_data_dir)

        dialog = QDialog()
        dialog.setWindowTitle("Events detection")
        dialog.setMinimumWidth(540)
        dialog.setFixedHeight(620)
        outer = QVBoxLayout(dialog)
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_widget = QWidget()
        form = QFormLayout(scroll_widget)

        group_combo, channels_list = self.build_group_channel_selector(
            form,
            groups,
            header,
            preferred_group_idx=int(common.get("group_idx", 0)),
            preferred_channels=common.get("channel_indexes", []),
        )

        pipeline_selector = PipelineSelector(
            self.pipelines_path(add_on_data_dir),
            selected_name=str(params.get("pipeline", "raw")),
        )
        form.addRow("Preprocessing pipeline:", pipeline_selector)

        mode_combo = QComboBox()
        mode_combo.addItem("TTL", MODE_TTL)
        mode_combo.addItem("Above threshold", MODE_ABOVE)
        saved_mode = str(params.get("mode", MODE_TTL))
        if saved_mode in {"digital", "analog"}:
            saved_mode = MODE_ABOVE if saved_mode == "digital" else MODE_TTL
        idx = mode_combo.findData(saved_mode)
        mode_combo.setCurrentIndex(max(0, idx))
        form.addRow("Detection mode:", mode_combo)

        name_edit = QLineEdit(str(params.get("event_name", "")))
        form.addRow("Event name:", name_edit)

        # TTL-specific
        edge_combo = QComboBox()
        edge_combo.addItem("Rising edge", EDGE_RISING)
        edge_combo.addItem("Falling edge", EDGE_FALLING)
        edge_idx = edge_combo.findData(str(params.get("edge", EDGE_RISING)))
        edge_combo.setCurrentIndex(max(0, edge_idx))
        form.addRow("TTL edge:", edge_combo)

        ttl_threshold_spin = QDoubleSpinBox()
        ttl_threshold_spin.setDecimals(6)
        ttl_threshold_spin.setRange(-1e9, 1e9)
        ttl_threshold_spin.setValue(float(params.get("ttl_threshold", params.get("height", 0.5))))
        form.addRow("TTL threshold:", ttl_threshold_spin)

        # Above-threshold specific
        height_spin = QDoubleSpinBox()
        height_spin.setDecimals(6)
        height_spin.setRange(-1e9, 1e9)
        height_spin.setValue(float(params.get("height", 50.0)))
        form.addRow("Height (threshold):", height_spin)

        distance_spin = QDoubleSpinBox()
        distance_spin.setDecimals(3)
        distance_spin.setRange(0.0, 1e9)
        distance_spin.setValue(float(params.get("distance_ms", 1.0)))
        form.addRow("Min distance (ms):", distance_spin)

        def _sync_mode_visibility() -> None:
            is_ttl = str(mode_combo.currentData()) == MODE_TTL
            edge_combo.setEnabled(is_ttl)
            ttl_threshold_spin.setEnabled(is_ttl)
            height_spin.setEnabled(not is_ttl)

        mode_combo.currentIndexChanged.connect(lambda _i: _sync_mode_visibility())
        _sync_mode_visibility()

        scroll_area.setWidget(scroll_widget)
        outer.addWidget(scroll_area)
        actions = QHBoxLayout()
        btn_cancel = QPushButton("Cancel")
        btn_run = QPushButton("Run")
        actions.addStretch(1)
        actions.addWidget(btn_cancel)
        actions.addWidget(btn_run)
        outer.addLayout(actions)
        btn_cancel.clicked.connect(dialog.reject)
        btn_run.clicked.connect(dialog.accept)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return None

        channels = self.selected_channels(channels_list)
        if not channels:
            QMessageBox.warning(dialog, "Events detection", "Select at least one channel.")
            return None
        event_name = name_edit.text().strip()
        if not event_name:
            QMessageBox.warning(dialog, "Events detection", "Event name must not be empty.")
            return None
        existing = {entry.name for entry in (session_manager.events_vocabulary or {}).values()}
        if event_name in existing:
            QMessageBox.warning(dialog, "Events detection", "Event name must be unique.")
            return None

        mode = str(mode_combo.currentData())
        self.save_common(
            add_on_data_dir,
            {"group_idx": int(group_combo.currentData()), "channel_indexes": channels},
        )
        payload = {
            "pipeline": pipeline_selector.current_pipeline_name(),
            "mode": mode,
            "event_name": event_name,
            "edge": str(edge_combo.currentData()),
            "ttl_threshold": float(ttl_threshold_spin.value()),
            "height": float(height_spin.value()),
            "distance_ms": float(distance_spin.value()),
        }
        self.save_params(add_on_data_dir, payload)
        return {"channels": channels, **payload}

    def run(self, session_manager, add_on_data_dir):
        header = session_manager.header
        add_on_data_dir = Path(add_on_data_dir)
        params = self._ask_parameters(session_manager, header, add_on_data_dir)
        if params is None:
            return

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
                None,
                "Events detection",
                "No events were detected. Vocabulary was not created.",
            )
            return

        n_detected = len(detected)
        if n_detected > CONFIRM_THRESHOLD:
            answer = QMessageBox.question(
                None,
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
