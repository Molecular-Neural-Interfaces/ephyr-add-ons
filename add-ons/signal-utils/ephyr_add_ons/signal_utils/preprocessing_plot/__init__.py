"""Preprocessing plot add-on.

Runnable. For the selected channels, renders one figure per channel with one
subplot per selected preprocessing pipeline stacked vertically, so different
pipelines can be compared side by side. A preprocessing progress screen is
shown while each pipeline runs.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from ephyr.core.add_ons.base import BaseAddOn
from ephyr.core.add_ons import (
    PipelineBuilderDialog,
    read_pipeline_store,
)
from ephyr.logger import ephyr_logger

from ephyr_add_ons.signal_utils._common import SignalUtilsBase


class PreprocessingPlotAddOn(SignalUtilsBase, BaseAddOn):
    TRANSFORMATION = False
    VIEWABLE = False
    RUNNABLE = True

    def _ask_parameters(self, session_manager, header, add_on_data_dir: Path):
        groups = self.ensure_non_aux_groups(session_manager, "Preprocessing plot")
        if groups is None:
            return None

        common = self.load_common(add_on_data_dir)
        params = self.load_params(add_on_data_dir)

        dialog = QDialog()
        dialog.setWindowTitle("Preprocessing plot")
        dialog.setMinimumWidth(560)
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

        pipelines_path = self.pipelines_path(add_on_data_dir)
        store = read_pipeline_store(pipelines_path)
        pipelines_list = QListWidget()
        pipelines_list.setSelectionMode(QListWidget.SelectionMode.MultiSelection)
        selected_pipelines = set(params.get("selected_pipelines", ["raw"]))

        def reload_pipelines(select_names=None) -> None:
            select = set(select_names) if select_names is not None else {
                pipelines_list.item(i).data(Qt.ItemDataRole.UserRole)
                for i in range(pipelines_list.count())
                if pipelines_list.item(i).isSelected()
            } or selected_pipelines
            pipelines_list.clear()
            for name in sorted(read_pipeline_store(pipelines_path).keys()):
                item = QListWidgetItem(name)
                item.setData(Qt.ItemDataRole.UserRole, name)
                item.setSelected(name in select)
                pipelines_list.addItem(item)

        reload_pipelines(selected_pipelines)
        form.addRow("Pipelines:", pipelines_list)
        manage_btn = QPushButton("Manage pipelines...")

        def open_manager() -> None:
            builder = PipelineBuilderDialog(pipelines_path, parent=dialog)
            if builder.exec() == QDialog.DialogCode.Accepted:
                saved = builder.saved_pipeline_name()
                reload_pipelines(({saved} if saved else None))

        manage_btn.clicked.connect(open_manager)
        form.addRow("", manage_btn)

        sweep_duration_ms = self.sweep_duration_ms(session_manager, header)
        default_from, default_to = self.default_window_ms(session_manager, header)
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
            QMessageBox.warning(dialog, "Preprocessing plot", "Select at least one channel.")
            return None
        pipeline_names = [
            pipelines_list.item(i).data(Qt.ItemDataRole.UserRole)
            for i in range(pipelines_list.count())
            if pipelines_list.item(i).isSelected()
        ]
        if not pipeline_names:
            QMessageBox.warning(dialog, "Preprocessing plot", "Select at least one pipeline.")
            return None
        window_from_ms = float(window_from_spin.value())
        window_to_ms = float(window_to_spin.value())
        if window_to_ms <= window_from_ms:
            QMessageBox.warning(dialog, "Preprocessing plot", "Window 'to' must be greater than 'from'.")
            return None

        selection = {
            "event_names": self.selected_names_from_list(events_list),
            "event_before_ms": float(before_spin.value()),
            "event_after_ms": float(after_spin.value()),
            "events_mode": self.selection_mode_from_combo(events_mode_combo),
            "period_names": self.selected_names_from_list(periods_list),
            "periods_mode": self.selection_mode_from_combo(periods_mode_combo),
        }

        self.save_common(
            add_on_data_dir,
            {
                "group_idx": int(group_combo.currentData()),
                "channel_indexes": channels,
                **self.event_period_selection_to_common(selection),
            },
        )
        self.save_params(
            add_on_data_dir,
            {
                "selected_pipelines": pipeline_names,
                "window_from_ms": window_from_ms,
                "window_to_ms": window_to_ms,
                "plot_image": plot_image_checkbox.isChecked(),
            },
        )
        return {
            "channels": channels,
            "pipeline_names": pipeline_names,
            "window_from_ms": window_from_ms,
            "window_to_ms": window_to_ms,
            "selection": selection,
            "plot_image": plot_image_checkbox.isChecked(),
        }

    def run(self, session_manager, add_on_data_dir):
        add_on_data_dir = Path(add_on_data_dir)
        header = session_manager.header
        params = self._ask_parameters(session_manager, header, add_on_data_dir)
        if params is None:
            return

        sweep_idx = int(session_manager.gui_setup.current_sweep_idx)
        sample_rate = float(header.sample_rate)
        start_sample, end_sample, start_second, end_second, n_samples = self.resolve_window_samples(
            session_manager, header, params["window_from_ms"], params["window_to_ms"]
        )

        channels = params["channels"]
        raw_matrix = self.channel_matrix_from_session(
            session_manager, channels, sweep_idx, start_sample, end_sample, sample_rate
        )

        valid_mask = self.build_selection_valid_mask(
            session_manager,
            n_samples=n_samples,
            sample_rate=sample_rate,
            sweep_idx=sweep_idx,
            start_second=start_second,
            end_second=end_second,
            selection=params["selection"],
        )

        store = read_pipeline_store(self.pipelines_path(add_on_data_dir))
        local_time = np.arange(n_samples, dtype=float) / sample_rate + start_second

        pipeline_names = params["pipeline_names"]
        processed_by_pipeline = {}
        n_pipelines = len(pipeline_names)
        for p_idx, name in enumerate(pipeline_names):
            pipeline = store.get(name)
            base = int((p_idx / max(1, n_pipelines)) * 90)
            span = int(90 / max(1, n_pipelines))
            processed = yield from self.iter_apply_pipeline(
                raw_matrix,
                sample_rate,
                pipeline,
                base_progress=base,
                progress_span=span,
                message_prefix=f"Preprocessing pipeline '{name}'",
            )
            processed_by_pipeline[name] = np.asarray(processed, dtype=np.float64)

        total = len(channels)
        yield {"progress": 90, "message": "Rendering preprocessing plots..."}
        for c_idx, channel_idx in enumerate(channels):
            channel_name = self.channel_name(header, int(channel_idx))
            fig, axes = plt.subplots(
                n_pipelines, 1, figsize=(14, max(3, 2.6 * n_pipelines)), sharex=True
            )
            if n_pipelines == 1:
                axes = [axes]
            for ax, name in zip(axes, pipeline_names):
                matrix = processed_by_pipeline[name]
                signal = matrix[c_idx].copy() if c_idx < matrix.shape[0] else np.zeros(n_samples)
                if valid_mask.size == signal.size:
                    signal[~valid_mask] = 0.0
                plot_len = min(signal.size, local_time.size)
                ax.plot(local_time[:plot_len], signal[:plot_len], linewidth=0.7, color="#1b4dff")
                ax.set_xlim(start_second, end_second)
                ax.set_ylabel("uV")
                ax.grid(True, alpha=0.25)
                ax.set_title(f"{channel_name} | pipeline: {name}")
            axes[-1].set_xlabel("Time (s)")
            fig.tight_layout()
            if params["plot_image"]:
                try:
                    plt.show()
                except Exception as e:
                    ephyr_logger().debug(str(e))
                    plt.close(fig)
            else:
                plt.close(fig)
            yield {"progress": int(90 + ((c_idx + 1) / total) * 10), "message": f"Plotted channel {c_idx + 1}/{total}"}
