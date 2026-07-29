"""Spectrogram plot add-on.

Runnable. For the selected channels (each on its own figure), draws an STFT
time-frequency spectrogram for the specified frequency range and time window.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
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
from scipy.signal import spectrogram

from ephyr.core.add_ons.base import BaseAddOn
from ephyr.core.add_ons import (
    PipelineSelector,
    read_pipeline_store,
)
from ephyr.logger import ephyr_logger

from ephyr_add_ons.signal_utils._common import SignalUtilsBase


class SpectrogramPlotAddOn(SignalUtilsBase, BaseAddOn):
    TRANSFORMATION = False
    VIEWABLE = False
    RUNNABLE = True

    def _ask_parameters(self, session_manager, header, add_on_data_dir: Path):
        groups = self.ensure_non_aux_groups(session_manager, "Spectrogram plot")
        if groups is None:
            return None
        common = self.load_common(add_on_data_dir)
        params = self.load_params(add_on_data_dir)

        dialog = QDialog()
        dialog.setWindowTitle("Spectrogram plot")
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
        form.addRow("Pipeline:", pipeline_selector)

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

        nyq = float(header.sample_rate) / 2.0
        freq_min_spin = QDoubleSpinBox()
        freq_min_spin.setRange(0.0, nyq)
        freq_min_spin.setDecimals(2)
        freq_min_spin.setValue(float(params.get("freq_min_hz", 1.0)))
        form.addRow("Freq min (Hz):", freq_min_spin)
        freq_max_spin = QDoubleSpinBox()
        freq_max_spin.setRange(1.0, nyq)
        freq_max_spin.setDecimals(2)
        freq_max_spin.setValue(min(float(params.get("freq_max_hz", min(300.0, nyq))), nyq))
        form.addRow("Freq max (Hz):", freq_max_spin)

        nperseg_spin = QSpinBox()
        nperseg_spin.setRange(16, 1_048_576)
        nperseg_spin.setValue(int(params.get("nperseg", 256)))
        form.addRow("STFT nperseg:", nperseg_spin)
        overlap_spin = QSpinBox()
        overlap_spin.setRange(0, 95)
        overlap_spin.setValue(int(params.get("overlap_pct", 75)))
        form.addRow("STFT overlap (%):", overlap_spin)

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
            QMessageBox.warning(dialog, "Spectrogram plot", "Select at least one channel.")
            return None
        window_from_ms = float(window_from_spin.value())
        window_to_ms = float(window_to_spin.value())
        if window_to_ms <= window_from_ms:
            QMessageBox.warning(dialog, "Spectrogram plot", "Window 'to' must be greater than 'from'.")
            return None
        freq_min = float(freq_min_spin.value())
        freq_max = float(freq_max_spin.value())
        if freq_max <= freq_min:
            QMessageBox.warning(dialog, "Spectrogram plot", "Freq max must be greater than freq min.")
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
                "pipeline": pipeline_selector.current_pipeline_name(),
                "window_from_ms": window_from_ms,
                "window_to_ms": window_to_ms,
                "freq_min_hz": freq_min,
                "freq_max_hz": freq_max,
                "nperseg": int(nperseg_spin.value()),
                "overlap_pct": int(overlap_spin.value()),
                "plot_image": plot_image_checkbox.isChecked(),
            },
        )
        return {
            "channels": channels,
            "pipeline": pipeline_selector.current_pipeline_name(),
            "window_from_ms": window_from_ms,
            "window_to_ms": window_to_ms,
            "freq_min_hz": freq_min,
            "freq_max_hz": freq_max,
            "nperseg": int(nperseg_spin.value()),
            "overlap_pct": int(overlap_spin.value()),
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

        store = read_pipeline_store(self.pipelines_path(add_on_data_dir))
        processed = yield from self.iter_apply_pipeline(
            raw_matrix, sample_rate, store.get(params["pipeline"]), base_progress=0, progress_span=60
        )
        processed = np.asarray(processed, dtype=np.float64)

        valid_mask = self.build_selection_valid_mask(
            session_manager,
            n_samples=n_samples,
            sample_rate=sample_rate,
            sweep_idx=sweep_idx,
            start_second=start_second,
            end_second=end_second,
            selection=params["selection"],
        )

        nperseg = max(16, min(int(params["nperseg"]), n_samples))
        noverlap = max(0, min(int(nperseg * float(params["overlap_pct"]) / 100.0), nperseg - 1))

        total = len(channels)
        yield {"progress": 65, "message": "Computing spectrograms..."}
        rendered = 0
        for c_idx, channel_idx in enumerate(channels):
            signal = processed[c_idx].copy() if c_idx < processed.shape[0] else np.zeros(n_samples)
            if valid_mask.size == signal.size:
                signal[~valid_mask] = 0.0
            channel_name = self.channel_name(header, int(channel_idx))
            fig, ax = plt.subplots(figsize=(13, 5))
            try:
                freqs, times, sxx = spectrogram(
                    signal, fs=sample_rate, nperseg=nperseg, noverlap=noverlap
                )
                band = (freqs >= params["freq_min_hz"]) & (freqs <= params["freq_max_hz"])
                sxx_db = 10.0 * np.log10(np.maximum(sxx[band], 1e-20))
                mesh = ax.pcolormesh(times + start_second, freqs[band], sxx_db, shading="auto", cmap="viridis")
                ax.set_ylim(params["freq_min_hz"], params["freq_max_hz"])
                fig.colorbar(mesh, ax=ax, label="Power (dB)")
            except Exception as e:
                ephyr_logger().debug(str(e))
                plt.close(fig)
                continue
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Frequency (Hz)")
            ax.set_title(f"{channel_name} | STFT | pipeline: {params['pipeline']}")
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
            yield {"progress": int(65 + ((c_idx + 1) / total) * 35), "message": f"Spectrogram {c_idx + 1}/{total}"}
        yield {"progress": 100, "message": f"Spectrograms rendered for {rendered} channel(s)"}
