"""Spike navigation add-on.

Runnable. Opens a non-modal window where the user picks a detection result set
and a channel, sees the spike count, and steps through spikes (arrows or a
number field). Selecting a spike recenters the signal view on it, mirroring the
event navigation behaviour.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ephyr.core.add_ons.base import BaseAddOn
from ephyr.logger import ephyr_logger

from ephyr_add_ons.spike_utils._common import SpikePoint, SpikeUtilsBase


class _SpikeNavigationWindow(QWidget):
    def __init__(self, add_on, session_manager, add_on_data_dir: Path):
        super().__init__()
        self._add_on = add_on
        self._session_manager = session_manager
        self._add_on_data_dir = Path(add_on_data_dir)
        self._spikes: List[SpikePoint] = []
        self._current_index: int = 0

        self.setWindowTitle("Spike navigation")
        self.setWindowFlags(Qt.WindowType.Window)
        self.setMinimumWidth(360)
        layout = QVBoxLayout(self)
        form = QFormLayout()

        self._dir_combo = QComboBox()
        for path in self._add_on.list_detection_result_dirs(self._add_on_data_dir):
            self._dir_combo.addItem(self._add_on.detection_result_label(path), str(path))
        form.addRow("Detection method:", self._dir_combo)

        self._channel_combo = QComboBox()
        form.addRow("Channel:", self._channel_combo)

        self._count_label = QLabel("No spikes")
        form.addRow("Spikes:", self._count_label)

        self._index_spin = QSpinBox()
        self._index_spin.setRange(0, 0)
        form.addRow("Go to spike #:", self._index_spin)
        layout.addLayout(form)

        nav_row = QHBoxLayout()
        self._prev_btn = QPushButton("< Prev")
        self._next_btn = QPushButton("Next >")
        self._go_btn = QPushButton("Go")
        nav_row.addWidget(self._prev_btn)
        nav_row.addWidget(self._next_btn)
        nav_row.addStretch(1)
        nav_row.addWidget(self._go_btn)
        layout.addLayout(nav_row)

        self._dir_combo.currentIndexChanged.connect(lambda _i: self._reload_channels())
        self._channel_combo.currentIndexChanged.connect(lambda _i: self._reload_spikes())
        self._prev_btn.clicked.connect(lambda: self._step(-1))
        self._next_btn.clicked.connect(lambda: self._step(1))
        self._go_btn.clicked.connect(self._go_to_index)

        self._reload_channels()

    def _current_sweep_idx(self) -> int:
        return int(self._session_manager.gui_setup.current_sweep_idx)

    def _load_payload(self):
        selected = self._dir_combo.currentData()
        if not selected:
            return None
        return self._add_on.read_spikes_payload(Path(str(selected)), self._current_sweep_idx())

    def _reload_channels(self) -> None:
        self._channel_combo.blockSignals(True)
        self._channel_combo.clear()
        payload = self._load_payload()
        header = getattr(self._session_manager, "header", None)
        if payload is not None:
            for ch in sorted(payload.spikes_by_channel.keys()):
                name = self._add_on.channel_name(header, int(ch)) if header is not None else f"ch{int(ch)}"
                self._channel_combo.addItem(f"{name} [{ch}]", int(ch))
        self._channel_combo.blockSignals(False)
        self._reload_spikes()

    def _reload_spikes(self) -> None:
        payload = self._load_payload()
        channel = self._channel_combo.currentData()
        self._spikes = []
        if payload is not None and channel is not None:
            self._spikes = list(payload.spikes_by_channel.get(int(channel), []))
        self._current_index = 0
        total = len(self._spikes)
        self._count_label.setText(f"{total} spike(s) on this channel / sweep")
        self._index_spin.setRange(0, max(0, total - 1))
        self._index_spin.setValue(0)
        if total:
            self._center_on(0)

    def _step(self, delta: int) -> None:
        if not self._spikes:
            return
        self._current_index = max(0, min(self._current_index + delta, len(self._spikes) - 1))
        self._index_spin.setValue(self._current_index)
        self._center_on(self._current_index)

    def _go_to_index(self) -> None:
        if not self._spikes:
            return
        self._current_index = max(0, min(int(self._index_spin.value()), len(self._spikes) - 1))
        self._center_on(self._current_index)

    def _center_on(self, index: int) -> None:
        if not (0 <= index < len(self._spikes)):
            return
        spike = self._spikes[index]
        try:
            header = self._session_manager.header
            sample_rate = float(header.sample_rate)
            duration_ms = float(self._session_manager.gui_setup.duration_ms)
            sweep_idx = self._current_sweep_idx()
            sweep_points = int(header.number_of_points_per_sweep[sweep_idx])
            target_samples = int((spike.time_ms / 1000.0) * sample_rate)
            half_window = int((duration_ms / 2000.0) * sample_rate)
            new_start = max(0, target_samples - half_window)
            max_start = max(0, sweep_points - int((duration_ms / 1000.0) * sample_rate))
            new_start = min(new_start, max_start)
            self._session_manager.set_start_point(int(new_start))
            self._count_label.setText(
                f"Spike {index + 1}/{len(self._spikes)} @ {spike.time_ms:.2f} ms"
            )
        except Exception as e:
            ephyr_logger().debug(str(e))


class SpikeNavigationAddOn(SpikeUtilsBase, BaseAddOn):
    TRANSFORMATION = False
    VIEWABLE = False
    RUNNABLE = True

    def __init__(self):
        self._nav_window: Optional[_SpikeNavigationWindow] = None

    def run(self, session_manager, add_on_data_dir):
        add_on_data_dir = Path(add_on_data_dir)
        if not self.list_detection_result_dirs(add_on_data_dir):
            from PyQt6.QtWidgets import QMessageBox

            QMessageBox.warning(None, "Spike navigation", "No detected spikes yet. Run Spike detection first.")
            return
        try:
            if self._nav_window is not None:
                self._nav_window.close()
        except Exception as e:
            ephyr_logger().debug(str(e))
        self._nav_window = _SpikeNavigationWindow(self, session_manager, add_on_data_dir)
        self._nav_window.show()
        self._nav_window.raise_()
        self._nav_window.activateWindow()
