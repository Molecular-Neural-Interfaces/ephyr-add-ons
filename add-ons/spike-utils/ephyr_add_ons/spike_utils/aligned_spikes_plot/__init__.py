"""Aligned spikes plot add-on.

Runnable. For the selected channels and time window, overlays all detected
spike waveforms (aligned on spike time) plus their mean, on one figure per
channel. Waveforms are extracted from the signal reprocessed with the pipeline
that was used for detection.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from ephyr.core.add_ons.base import BaseAddOn
from ephyr.core.add_ons import read_pipeline_store
from ephyr.logger import ephyr_logger

from ephyr_add_ons.spike_utils._common import SpikeUtilsBase


class AlignedSpikesPlotAddOn(SpikeUtilsBase, BaseAddOn):
    TRANSFORMATION = False
    VIEWABLE = False
    RUNNABLE = True

    def _ask_parameters(self, session_manager, header, add_on_data_dir: Path) -> Optional[dict]:
        params = self.load_params(add_on_data_dir)
        selected_dir = self.choose_result_dir_dialog(
            "Aligned spikes plot",
            add_on_data_dir,
            selected_dir=str(params.get("selected_dir", "")),
            label="Detection method:",
        )
        if selected_dir is None:
            return None

        sweep_idx = int(session_manager.gui_setup.current_sweep_idx)
        payload = self.read_spikes_payload(selected_dir, sweep_idx)
        if payload is None:
            QMessageBox.warning(
                None,
                "Aligned spikes plot",
                "No spikes for current sweep in selected detection method.",
            )
            return None

        detection_channels = self.channels_for_detection_payload(
            payload,
            channel_groups=session_manager.gui_setup.channels_groups,
        )
        if not detection_channels:
            detection_channels = sorted(int(ch) for ch in payload.spikes_by_channel.keys())
        if not detection_channels:
            QMessageBox.warning(None, "Aligned spikes plot", "Selected detection has no channels.")
            return None

        dialog = QDialog()
        dialog.setWindowTitle("Aligned spikes plot")
        dialog.setMinimumWidth(520)
        dialog.setFixedHeight(560)
        outer = QVBoxLayout(dialog)
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_widget = QWidget()
        form = QFormLayout(scroll_widget)

        meta = self.read_detection_meta(selected_dir)
        group_label = QLabel(
            (payload.group_name or meta.group_name or "Group").strip() or "Group"
        )
        form.addRow("Detection group:", group_label)

        channels_list = QListWidget()
        channels_list.setSelectionMode(QListWidget.SelectionMode.MultiSelection)
        preferred = set(int(c) for c in (params.get("channel_indexes") or []))
        select_all = not preferred
        for ch_idx in detection_channels:
            ch_name = self.channel_name(header, int(ch_idx))
            item = QListWidgetItem(f"{ch_idx} [{ch_name}]")
            item.setData(Qt.ItemDataRole.UserRole, int(ch_idx))
            item.setSelected(select_all or int(ch_idx) in preferred)
            channels_list.addItem(item)
        form.addRow("Channels:", channels_list)

        sample_rate = float(header.sample_rate)
        sweep_points = int(header.number_of_points_per_sweep[sweep_idx])
        sweep_duration_ms = (sweep_points / sample_rate) * 1000.0 if sample_rate > 0 else 0.0
        default_from = float(session_manager.gui_setup.start_point) * 1000.0 / sample_rate if sample_rate > 0 else 0.0
        default_to = default_from + float(session_manager.gui_setup.duration_ms)

        window_from_spin = QDoubleSpinBox()
        window_from_spin.setRange(0.0, max(0.0, sweep_duration_ms))
        window_from_spin.setDecimals(3)
        window_from_spin.setValue(max(0.0, min(float(params.get("window_from_ms", default_from)), sweep_duration_ms)))
        form.addRow("Window from (ms):", window_from_spin)
        window_to_spin = QDoubleSpinBox()
        window_to_spin.setRange(0.0, max(0.0, sweep_duration_ms))
        window_to_spin.setDecimals(3)
        window_to_spin.setValue(max(0.0, min(float(params.get("window_to_ms", default_to)), sweep_duration_ms)))
        form.addRow("Window to (ms):", window_to_spin)

        pre_spin = QDoubleSpinBox()
        pre_spin.setRange(0.1, 100.0)
        pre_spin.setDecimals(3)
        pre_spin.setValue(float(params.get("pre_ms", 1.0)))
        form.addRow("Pre-spike (ms):", pre_spin)
        post_spin = QDoubleSpinBox()
        post_spin.setRange(0.1, 100.0)
        post_spin.setDecimals(3)
        post_spin.setValue(float(params.get("post_ms", 2.0)))
        form.addRow("Post-spike (ms):", post_spin)

        middle_line = QCheckBox("Middle line")
        middle_line.setChecked(bool(params.get("middle_line", True)))
        form.addRow("Options:", middle_line)
        grid_checkbox = QCheckBox("Background grid")
        grid_checkbox.setChecked(bool(params.get("grid", True)))
        form.addRow("", grid_checkbox)
        plot_image_checkbox = QCheckBox("Plot image")
        plot_image_checkbox.setChecked(bool(params.get("plot_image", True)))
        form.addRow("", plot_image_checkbox)

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
            QMessageBox.warning(dialog, "Aligned spikes plot", "Select at least one channel.")
            return None
        window_from_ms = float(window_from_spin.value())
        window_to_ms = float(window_to_spin.value())
        if window_to_ms <= window_from_ms:
            QMessageBox.warning(dialog, "Aligned spikes plot", "Window 'to' must be greater than 'from'.")
            return None

        self.save_params(
            add_on_data_dir,
            {
                "selected_dir": str(selected_dir),
                "channel_indexes": channels,
                "window_from_ms": window_from_ms,
                "window_to_ms": window_to_ms,
                "pre_ms": float(pre_spin.value()),
                "post_ms": float(post_spin.value()),
                "middle_line": middle_line.isChecked(),
                "grid": grid_checkbox.isChecked(),
                "plot_image": plot_image_checkbox.isChecked(),
            },
        )
        return {
            "channels": channels,
            "selected_dir": Path(selected_dir),
            "window_from_ms": window_from_ms,
            "window_to_ms": window_to_ms,
            "pre_ms": float(pre_spin.value()),
            "post_ms": float(post_spin.value()),
            "middle_line": middle_line.isChecked(),
            "grid": grid_checkbox.isChecked(),
            "plot_image": plot_image_checkbox.isChecked(),
        }

    def run(self, session_manager, add_on_data_dir):
        header = session_manager.header
        add_on_data_dir = Path(add_on_data_dir)
        params = self._ask_parameters(session_manager, header, add_on_data_dir)
        if params is None:
            return

        sweep_idx = int(session_manager.gui_setup.current_sweep_idx)
        payload = self.read_spikes_payload(params["selected_dir"], sweep_idx)
        if payload is None:
            QMessageBox.warning(None, "Aligned spikes plot", "No spikes for current sweep in selected set.")
            return

        sample_rate = float(header.sample_rate)
        sweep_points = int(header.number_of_points_per_sweep[sweep_idx])
        channels = params["channels"]
        matrix = self.channel_matrix_from_session(
            session_manager, channels, sweep_idx, 0, sweep_points, sample_rate
        )
        store = read_pipeline_store(self.pipelines_path(add_on_data_dir))
        processed = yield from self.iter_apply_pipeline(
            matrix, sample_rate, store.get(payload.preprocessing_pipeline), base_progress=0, progress_span=40
        )
        processed = np.asarray(processed, dtype=np.float64)

        window_start_s = params["window_from_ms"] / 1000.0
        window_end_s = params["window_to_ms"] / 1000.0
        pre_samples = max(1, int(round(params["pre_ms"] * sample_rate / 1000.0)))
        post_samples = max(1, int(round(params["post_ms"] * sample_rate / 1000.0)))

        total = len(channels)
        rendered = 0
        for row_idx, channel_idx in enumerate(channels):
            signal = processed[row_idx].copy() if row_idx < processed.shape[0] else np.zeros(sweep_points)
            channel_name = self.channel_name(header, int(channel_idx))
            spikes = payload.spikes_by_channel.get(int(channel_idx), [])
            waveforms = []
            for spike in spikes:
                spike_time_s = float(spike.time_ms) / 1000.0
                if spike_time_s < window_start_s or spike_time_s > window_end_s:
                    continue
                center = int(round(spike_time_s * sample_rate))
                s0 = center - pre_samples
                s1 = center + post_samples
                if s0 < 0 or s1 >= signal.size:
                    continue
                waveforms.append(signal[s0:s1])
            if not waveforms:
                yield {"progress": int(40 + ((row_idx + 1) / total) * 60), "message": f"No waveforms for channel {channel_idx}"}
                continue
            waveforms_arr = np.asarray(waveforms, dtype=np.float64)
            time_ms = np.linspace(-params["pre_ms"], params["post_ms"], waveforms_arr.shape[1])
            mean_waveform = np.mean(waveforms_arr, axis=0)

            fig, ax = plt.subplots(figsize=(8, 4))
            for waveform in waveforms_arr:
                ax.plot(time_ms, waveform, color="grey", alpha=0.2, linewidth=0.6)
            ax.plot(time_ms, mean_waveform, color="black", linewidth=1.6)
            if params["middle_line"]:
                ax.axvline(0.0, color="blue", linestyle="--", linewidth=0.8)
            ax.set_title(f"Aligned spikes {channel_name} (n={waveforms_arr.shape[0]})")
            ax.set_xlabel("Time (ms)")
            ax.set_ylabel("Amplitude (uV)")
            if params["grid"]:
                ax.grid(True, alpha=0.25)
            fig.tight_layout()
            if params["plot_image"]:
                try:
                    plt.show()
                except Exception as e:
                    ephyr_logger().debug(str(e))
                    plt.close(fig)
            else:
                plt.close(fig)
            rendered += 1
            yield {"progress": int(40 + ((row_idx + 1) / total) * 60), "message": f"Plotted channel {row_idx + 1}/{total}"}
        yield {"progress": 100, "message": f"Aligned spikes rendered for {rendered} channel(s)"}
