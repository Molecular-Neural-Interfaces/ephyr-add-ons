"""Spike detection add-on.

Runnable. Detects spikes on the full current sweep for the selected channels
using a configurable preprocessing pipeline and MAD-based detection (global or
adaptive rolling sigma). Events and periods can be ignored. Results are stored
per sweep so the viewer / navigation / aligned / raster add-ons can read them.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np
from PyQt6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ephyr.core.add_ons.base import BaseAddOn
from ephyr.core.add_ons import (
    PipelineSelector,
    read_pipeline_store,
)
from ephyr.logger import ephyr_logger

from ephyr_add_ons.spike_utils._common import (
    DetectionResultMeta,
    SpikePoint,
    SpikesPayload,
    SpikeUtilsBase,
    detect_spikes_adaptive_mad,
    detect_spikes_mad,
    merge_spikes_global,
    rolling_sigma_mad,
    safe_detection_dir_name,
)


class SpikeDetectionAddOn(SpikeUtilsBase, BaseAddOn):
    TRANSFORMATION = False
    VIEWABLE = False
    RUNNABLE = True

    def _ask_parameters(self, session_manager, header, add_on_data_dir: Path) -> Optional[dict]:
        groups = self.ensure_non_aux_groups(session_manager, "Spike detection")
        if groups is None:
            return None
        common = self.load_common(add_on_data_dir)
        params = self.load_params(add_on_data_dir)

        dialog = QDialog()
        dialog.setWindowTitle("Spike detection")
        dialog.setMinimumWidth(560)
        dialog.setFixedHeight(640)
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

        threshold_spin = QDoubleSpinBox()
        threshold_spin.setRange(0.0, 1_000_000.0)
        threshold_spin.setDecimals(6)
        threshold_spin.setSingleStep(0.1)
        threshold_spin.setValue(float(params.get("threshold", 6.0)))
        form.addRow("Threshold multiplier:", threshold_spin)

        adaptive_checkbox = QCheckBox("Use adaptive rolling sigma")
        adaptive_checkbox.setChecked(bool(params.get("adaptive_sigma", True)))
        form.addRow("Sigma:", adaptive_checkbox)

        sigma_window = QDoubleSpinBox()
        sigma_window.setRange(1.0, 60_000.0)
        sigma_window.setDecimals(3)
        sigma_window.setValue(float(params.get("sigma_window_ms", 500.0)))
        form.addRow("Sigma window (ms):", sigma_window)
        sigma_step = QDoubleSpinBox()
        sigma_step.setRange(1.0, 60_000.0)
        sigma_step.setDecimals(3)
        sigma_step.setValue(float(params.get("sigma_step_ms", 100.0)))
        form.addRow("Sigma step (ms):", sigma_step)
        sigma_floor = QDoubleSpinBox()
        sigma_floor.setRange(0.0, 1_000_000.0)
        sigma_floor.setDecimals(6)
        sigma_floor.setValue(float(params.get("sigma_floor_uv", 2.0)))
        form.addRow("Sigma floor (uV):", sigma_floor)
        sigma_smooth = QSpinBox()
        sigma_smooth.setRange(1, 1000)
        sigma_smooth.setValue(int(params.get("sigma_smooth_windows", 3)))
        form.addRow("Sigma smooth windows:", sigma_smooth)

        detect_negative = QCheckBox("Detect downward spikes")
        detect_negative.setChecked(bool(params.get("detect_negative", True)))
        form.addRow("Polarity:", detect_negative)
        detect_positive = QCheckBox("Detect upward spikes")
        detect_positive.setChecked(bool(params.get("detect_positive", False)))
        form.addRow("", detect_positive)

        merge_window = QDoubleSpinBox()
        merge_window.setRange(0.0, 1000.0)
        merge_window.setDecimals(3)
        merge_window.setValue(float(params.get("merge_window_ms", 1.0)))
        form.addRow("Merge window (ms):", merge_window)

        ep_sel = self.load_event_period_selection(common)
        events_list, before_spin, after_spin, events_mode_combo = self.build_ignore_events_controls(
            form,
            session_manager,
            selected_names=ep_sel["event_names"],
            before_ms=ep_sel["event_before_ms"],
            after_ms=ep_sel["event_after_ms"],
            selection_mode=ep_sel["events_mode"],
        )
        periods_list, periods_mode_combo = self.build_ignore_periods_controls(
            form,
            session_manager,
            selected_names=ep_sel["period_names"],
            selection_mode=ep_sel["periods_mode"],
        )

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
            QMessageBox.warning(dialog, "Spike detection", "Select at least one channel.")
            return None
        if not detect_negative.isChecked() and not detect_positive.isChecked():
            QMessageBox.warning(dialog, "Spike detection", "Select at least one polarity.")
            return None

        group_idx = int(group_combo.currentData())
        group = next((g for idx, g in groups if idx == group_idx), None)
        group_name = str(getattr(group, "name", "") or "").strip() or f"Group {group_idx}"
        group_key = self.group_stable_key(group) if group is not None else ""

        selection = {
            "event_names": self.selected_names_from_list(events_list),
            "event_before_ms": float(before_spin.value()),
            "event_after_ms": float(after_spin.value()),
            "events_mode": self.selection_mode_from_combo(events_mode_combo),
            "period_names": self.selected_names_from_list(periods_list),
            "periods_mode": self.selection_mode_from_combo(periods_mode_combo),
        }
        sigma_params = {
            "window_ms": float(sigma_window.value()),
            "step_ms": float(sigma_step.value()),
            "sigma_floor_uv": float(sigma_floor.value()),
            "smooth_windows": int(sigma_smooth.value()),
        }

        self.save_common(
            add_on_data_dir,
            {
                "group_idx": group_idx,
                "channel_indexes": channels,
                **self.event_period_selection_to_common(selection),
            },
        )
        self.save_params(
            add_on_data_dir,
            {
                "pipeline": pipeline_selector.current_pipeline_name(),
                "threshold": float(threshold_spin.value()),
                "adaptive_sigma": adaptive_checkbox.isChecked(),
                "detect_negative": detect_negative.isChecked(),
                "detect_positive": detect_positive.isChecked(),
                "merge_window_ms": float(merge_window.value()),
                "sigma_window_ms": sigma_params["window_ms"],
                "sigma_step_ms": sigma_params["step_ms"],
                "sigma_floor_uv": sigma_params["sigma_floor_uv"],
                "sigma_smooth_windows": sigma_params["smooth_windows"],
            },
        )
        return {
            "channels": channels,
            "group_key": group_key,
            "group_name": group_name,
            "pipeline": pipeline_selector.current_pipeline_name(),
            "threshold": float(threshold_spin.value()),
            "adaptive_sigma": adaptive_checkbox.isChecked(),
            "detect_negative": detect_negative.isChecked(),
            "detect_positive": detect_positive.isChecked(),
            "merge_window_ms": float(merge_window.value()),
            "sigma_params": sigma_params,
            "selection": selection,
        }

    def run(self, session_manager, add_on_data_dir):
        header = session_manager.header
        add_on_data_dir = Path(add_on_data_dir)
        params = self._ask_parameters(session_manager, header, add_on_data_dir)
        if params is None:
            return

        sweep_idx = int(session_manager.gui_setup.current_sweep_idx)
        sample_rate = float(header.sample_rate)
        sweep_points = int(header.number_of_points_per_sweep[sweep_idx])
        channels = params["channels"]
        matrix = self.channel_matrix_from_session(
            session_manager, channels, sweep_idx, 0, sweep_points, sample_rate
        )

        store = read_pipeline_store(self.pipelines_path(add_on_data_dir))
        processed = yield from self.iter_apply_pipeline(
            matrix, sample_rate, store.get(params["pipeline"]), base_progress=0, progress_span=40
        )
        processed = np.asarray(processed, dtype=np.float64)

        end_second = sweep_points / sample_rate
        selection = params["selection"]
        valid_mask = self.build_selection_valid_mask(
            session_manager,
            n_samples=sweep_points,
            sample_rate=sample_rate,
            sweep_idx=sweep_idx,
            start_second=0.0,
            end_second=end_second,
            selection=selection,
        )

        threshold = params["threshold"]
        adaptive_sigma = params["adaptive_sigma"]
        merge_window_ms = params["merge_window_ms"]
        detect_negative = params["detect_negative"]
        detect_positive = params["detect_positive"]
        sigma_params = params["sigma_params"]
        sigma_floor = float(sigma_params["sigma_floor_uv"])

        spikes_by_channel = {}
        total = len(channels)
        for row_idx, channel_idx in enumerate(channels):
            signal = np.asarray(processed[row_idx], dtype=np.float64) if row_idx < processed.shape[0] else np.zeros(sweep_points)
            sigma_t = None
            if adaptive_sigma:
                sigma_t = rolling_sigma_mad(
                    signal,
                    sample_rate,
                    window_ms=float(sigma_params["window_ms"]),
                    step_ms=float(sigma_params["step_ms"]),
                    sigma_floor_uv=sigma_floor,
                    smooth_windows=int(sigma_params["smooth_windows"]),
                    mask=valid_mask if valid_mask.size == signal.size else None,
                )
            found = []
            for polarity_positive in ([False] if detect_negative else []) + ([True] if detect_positive else []):
                if adaptive_sigma:
                    found.extend(
                        detect_spikes_adaptive_mad(
                            signal, sample_rate, threshold, merge_window_ms, polarity_positive, sigma_t, sigma_floor
                        )
                    )
                else:
                    found.extend(
                        detect_spikes_mad(signal, sample_rate, threshold, merge_window_ms, polarity_positive, sigma_floor)
                    )
            if valid_mask.size == signal.size:
                found = [sp for sp in found if 0 <= int(sp.sample_idx) < valid_mask.size and valid_mask[int(sp.sample_idx)]]
            merged = (
                merge_spikes_global(found, sample_rate, min_distance_ms=merge_window_ms)
                if detect_negative and detect_positive
                else sorted(found, key=lambda sp: int(sp.sample_idx))
            )
            spikes_by_channel[int(channel_idx)] = [
                SpikePoint(
                    sample_idx=int(sp.sample_idx),
                    time_ms=float(sp.sample_idx) / sample_rate * 1000.0,
                    value=float(sp.value),
                    polarity=str(sp.polarity),
                )
                for sp in merged
            ]
            yield {"progress": int(40 + ((row_idx + 1) / total) * 55), "message": f"Detected on channel {row_idx + 1}/{total}"}

        out_dir = add_on_data_dir / safe_detection_dir_name(
            params["pipeline"],
            threshold,
            adaptive_sigma,
            group_name=params["group_name"],
        )
        payload = SpikesPayload(
            detector_name="adaptive_mad" if adaptive_sigma else "mad",
            preprocessing_pipeline=params["pipeline"],
            threshold=float(threshold),
            sweep_idx=sweep_idx,
            sample_rate=sample_rate,
            adaptive_sigma=adaptive_sigma,
            sigma_params=sigma_params,
            detect_positive=detect_positive,
            detect_negative=detect_negative,
            merge_window_ms=merge_window_ms,
            group_key=str(params.get("group_key", "") or ""),
            group_name=str(params.get("group_name", "") or ""),
            ignore_event_names=list(selection.get("event_names") or []),
            ignore_before_ms=float(selection.get("event_before_ms", 0.0)),
            ignore_after_ms=float(selection.get("event_after_ms", 0.0)),
            ignore_period_names=list(selection.get("period_names") or []),
            events_mode=str(selection.get("events_mode", "ignore")),
            periods_mode=str(selection.get("periods_mode", "ignore")),
            spikes_by_channel=spikes_by_channel,
        )
        output_path = out_dir / f"{sweep_idx}.spikes.json"
        try:
            self.save_spikes_payload(output_path, payload)
            self.save_detection_meta(
                out_dir,
                DetectionResultMeta(
                    group_key=payload.group_key,
                    group_name=payload.group_name,
                    preprocessing_pipeline=payload.preprocessing_pipeline,
                    threshold=payload.threshold,
                    adaptive_sigma=payload.adaptive_sigma,
                    detector_name=payload.detector_name,
                ),
            )
        except Exception as e:
            ephyr_logger().debug(str(e))
            yield {"progress": 100, "message": "Failed to save spikes"}
            return
        n_total = sum(len(v) for v in spikes_by_channel.values())
        yield {"progress": 100, "message": f"Saved {n_total} spikes to {output_path.parent.name}"}
