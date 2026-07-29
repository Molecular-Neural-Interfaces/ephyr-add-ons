"""Raster plot add-on.

Runnable. Draws a horizontal raster (time on the x-axis, channels on the y-axis)
for a chosen time window, independent of the on-screen channel layout.
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
from ephyr.logger import ephyr_logger

from ephyr_add_ons.spike_utils._common import SpikeUtilsBase


class RasterPlotAddOn(SpikeUtilsBase, BaseAddOn):
    TRANSFORMATION = False
    VIEWABLE = False
    RUNNABLE = True

    def _ask_parameters(self, session_manager, header, add_on_data_dir: Path) -> Optional[dict]:
        params = self.load_params(add_on_data_dir)
        selected_dir = self.choose_result_dir_dialog(
            "Raster plot",
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
                "Raster plot",
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
            QMessageBox.warning(None, "Raster plot", "Selected detection has no channels.")
            return None

        dialog = QDialog()
        dialog.setWindowTitle("Raster plot")
        dialog.setMinimumWidth(520)
        dialog.setFixedHeight(520)
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

        plot_image_checkbox = QCheckBox("Plot image")
        plot_image_checkbox.setChecked(bool(params.get("plot_image", True)))
        form.addRow("Output:", plot_image_checkbox)

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
            QMessageBox.warning(dialog, "Raster plot", "Select at least one channel.")
            return None
        window_from_ms = float(window_from_spin.value())
        window_to_ms = float(window_to_spin.value())
        if window_to_ms <= window_from_ms:
            QMessageBox.warning(dialog, "Raster plot", "Window 'to' must be greater than 'from'.")
            return None

        self.save_params(
            add_on_data_dir,
            {
                "selected_dir": str(selected_dir),
                "channel_indexes": channels,
                "window_from_ms": window_from_ms,
                "window_to_ms": window_to_ms,
                "plot_image": plot_image_checkbox.isChecked(),
            },
        )
        return {
            "channels": channels,
            "selected_dir": Path(selected_dir),
            "window_from_ms": window_from_ms,
            "window_to_ms": window_to_ms,
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
            QMessageBox.warning(None, "Raster plot", "No spikes for current sweep in selected set.")
            return

        yield {"progress": 40, "message": "Building raster..."}
        channels: List[int] = params["channels"]
        labels = [self.channel_name(header, int(ch)) for ch in channels]
        start_s = params["window_from_ms"] / 1000.0
        end_s = params["window_to_ms"] / 1000.0

        xs: List[float] = []
        ys: List[float] = []
        for row_idx, channel_idx in enumerate(channels):
            for spike in payload.spikes_by_channel.get(int(channel_idx), []):
                t = float(spike.time_ms) / 1000.0
                if start_s <= t <= end_s:
                    xs.append(t)
                    ys.append(row_idx)

        n_channels = max(1, len(channels))
        fig, ax = plt.subplots(figsize=(14, max(4, 0.35 * n_channels + 2)))
        if xs:
            ax.plot(np.asarray(xs), np.asarray(ys), "|", color="black", markersize=10)
        ax.set_xlim(start_s, end_s)
        ax.set_ylim(-0.5, n_channels - 0.5)
        ax.invert_yaxis()  # channel 0 at top, ascending downward
        ax.set_yticks(list(range(len(channels))))
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Channel")
        ax.grid(True, axis="x", alpha=0.3)
        ax.set_title(
            f"Raster plot | {self.detection_result_label(params['selected_dir'])} | sweep {sweep_idx}"
        )
        fig.tight_layout()
        if params["plot_image"]:
            try:
                plt.show()
            except Exception as e:
                ephyr_logger().debug(str(e))
                plt.close(fig)
        else:
            plt.close(fig)
        yield {"progress": 100, "message": f"Raster plotted ({len(xs)} spikes)"}
