"""Non-modal video sync window: frame display tied to signal panel scrubbing."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Protocol, Sequence

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ephyr.logger import ephyr_logger

from ephyr_add_ons.labeling_utils.video_sync.detection import (
    MatchCandidate,
    TtlTrain,
    VideoFileFrameSource,
    VideoMeta,
    read_frame_bgr,
)
from ephyr_add_ons.labeling_utils.video_sync.mapping import TimeMapping, mapping_path


class FrameSource(Protocol):
    def read_bgr(self, frame_index: int): ...

    def close(self) -> None: ...


def _bgr_to_qpixmap(frame_bgr, max_width: int = 640) -> Optional[QPixmap]:
    if frame_bgr is None:
        return None
    h, w = frame_bgr.shape[:2]
    if frame_bgr.ndim == 2:
        rgb = np.stack([frame_bgr] * 3, axis=-1)
    else:
        rgb = frame_bgr[:, :, ::-1].copy()
    bytes_per_line = 3 * w
    image = QImage(rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888).copy()
    pix = QPixmap.fromImage(image)
    if pix.width() > max_width:
        pix = pix.scaledToWidth(max_width, Qt.TransformationMode.SmoothTransformation)
    return pix


class VideoSyncWindow(QWidget):
    """Live video/imaging frame synchronized to the center of the signal panel window."""

    def __init__(
        self,
        session_manager,
        add_on_data_dir: Path,
        video_meta: VideoMeta,
        trains: Sequence[TtlTrain],
        flash_times_ms: Sequence[float],
        candidates: Sequence[MatchCandidate],
        initial_mapping: Optional[TimeMapping] = None,
        selected_train_id: Optional[int] = None,
        signal_duration_ms: float = 0.0,
        frame_source: Optional[FrameSource] = None,
    ):
        super().__init__()
        self._session_manager = session_manager
        self._add_on_data_dir = Path(add_on_data_dir)
        self._meta = video_meta
        self._is_nwb = bool(video_meta.is_nwb)
        self._frame_source: FrameSource = frame_source or VideoFileFrameSource(video_meta.path)
        self._trains: List[TtlTrain] = list(trains)
        self._flash_times_ms = np.asarray(list(flash_times_ms), dtype=np.float64)
        self._candidates = list(candidates)
        self._signal_duration_ms = float(signal_duration_ms)
        self._mapping = initial_mapping or TimeMapping()
        self._last_frame_index: Optional[int] = None
        self._updating_ui = False

        title = "NWB imaging sync" if self._is_nwb else "Video sync"
        if self._is_nwb and self._meta.series_location:
            title = f"NWB imaging sync — {self._meta.series_location}"
        self.setWindowTitle(title)
        self.setWindowFlags(Qt.WindowType.Window)
        self.setMinimumWidth(520)
        self.setMinimumHeight(480)

        layout = QVBoxLayout(self)

        self._banner = QLabel("")
        self._banner.setWordWrap(True)
        self._banner.setStyleSheet("color: #a65c00;")
        layout.addWidget(self._banner)

        self._frame_label = QLabel("No frame")
        self._frame_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._frame_label.setMinimumHeight(280)
        self._frame_label.setStyleSheet("background: #1a1a1a; color: #ccc;")
        layout.addWidget(self._frame_label)

        self._status = QLabel("")
        self._status.setWordWrap(True)
        layout.addWidget(self._status)

        scrub_row = QHBoxLayout()
        scrub_row.addWidget(QLabel("Frame:" if self._is_nwb else "Video frame:"))
        self._frame_spin = QSpinBox()
        self._frame_spin.setRange(0, max(0, int(self._meta.frame_count) - 1))
        scrub_row.addWidget(self._frame_spin)
        scrub_row.addWidget(QLabel("Imaging ms:" if self._is_nwb else "Video ms:"))
        self._video_ms_spin = QDoubleSpinBox()
        self._video_ms_spin.setDecimals(1)
        self._video_ms_spin.setRange(0.0, max(1.0, float(self._meta.duration_ms)))
        self._video_ms_spin.setSingleStep(100.0)
        scrub_row.addWidget(self._video_ms_spin)
        self._follow_signal = True
        self._unfollow_btn = QPushButton("Unlock frame scrub")
        self._unfollow_btn.setCheckable(True)
        scrub_row.addWidget(self._unfollow_btn)
        layout.addLayout(scrub_row)

        self._train_label = QLabel("TTL sequence ↔ flashes:")
        self._train_combo = QComboBox()
        for train in self._trains:
            self._train_combo.addItem(train.label(), int(train.train_id))
        layout.addWidget(self._train_label)
        layout.addWidget(self._train_combo)

        self._rank_label: Optional[QLabel] = None
        if self._candidates:
            rank_bits = ", ".join(
                f"train {c.train_id} score={c.score:.2f}" for c in self._candidates[:5]
            )
            self._rank_label = QLabel(f"Auto-match ranking: {rank_bits}")
            layout.addWidget(self._rank_label)

        self._pairs_label = QLabel(
            "Sync pairs (signal_ms → imaging_ms):"
            if self._is_nwb
            else "Sync pairs (signal_ms → video_ms):"
        )
        self._pairs_list = QListWidget()
        layout.addWidget(self._pairs_label)
        layout.addWidget(self._pairs_list)

        pair_row = QHBoxLayout()
        self._add_pair_btn = QPushButton("Add pair (window center ↔ frame)")
        self._remove_pair_btn = QPushButton("Remove selected")
        pair_row.addWidget(self._add_pair_btn)
        pair_row.addWidget(self._remove_pair_btn)
        layout.addLayout(pair_row)

        nav_row = QHBoxLayout()
        self._prev_ttl_btn = QPushButton("< TTL")
        self._next_ttl_btn = QPushButton("TTL >")
        self._prev_flash_btn = QPushButton("< Flash")
        self._next_flash_btn = QPushButton("Flash >")
        nav_row.addWidget(self._prev_ttl_btn)
        nav_row.addWidget(self._next_ttl_btn)
        nav_row.addStretch(1)
        nav_row.addWidget(self._prev_flash_btn)
        nav_row.addWidget(self._next_flash_btn)
        layout.addLayout(nav_row)

        if self._is_nwb:
            self._train_label.hide()
            self._train_combo.hide()
            if self._rank_label is not None:
                self._rank_label.hide()
            self._prev_ttl_btn.hide()
            self._next_ttl_btn.hide()
            self._prev_flash_btn.hide()
            self._next_flash_btn.hide()

        # Select initial train
        pick_id = selected_train_id
        if pick_id is None and self._candidates:
            pick_id = int(self._candidates[0].train_id)
        if pick_id is not None:
            idx = self._train_combo.findData(int(pick_id))
            if idx >= 0:
                self._train_combo.setCurrentIndex(idx)

        if not self._is_nwb:
            if not self._mapping.pairs and self._candidates:
                best = self._candidates[0]
                self._mapping.set_pairs(best.pairs)
            elif not self._mapping.pairs:
                train = self._current_train()
                if train is not None and self._flash_times_ms.size:
                    from ephyr_add_ons.labeling_utils.video_sync.detection import align_sequences

                    self._mapping.set_pairs(
                        align_sequences(train.times_ms.tolist(), self._flash_times_ms.tolist())
                    )

        self._train_combo.currentIndexChanged.connect(self._on_train_changed)
        self._add_pair_btn.clicked.connect(self._add_pair_from_current)
        self._remove_pair_btn.clicked.connect(self._remove_selected_pair)
        self._prev_ttl_btn.clicked.connect(lambda: self._step_ttl(-1))
        self._next_ttl_btn.clicked.connect(lambda: self._step_ttl(1))
        self._prev_flash_btn.clicked.connect(lambda: self._step_flash(-1))
        self._next_flash_btn.clicked.connect(lambda: self._step_flash(1))
        self._unfollow_btn.toggled.connect(self._on_unlock_toggled)
        self._frame_spin.valueChanged.connect(self._on_frame_spin_changed)
        self._video_ms_spin.valueChanged.connect(self._on_video_ms_spin_changed)

        try:
            self._session_manager.start_point_changed.connect(self._on_time_changed)
            self._session_manager.duration_ms_changed.connect(self._on_time_changed)
            self._session_manager.current_sweep_idx_changed.connect(self._on_sweep_changed)
        except Exception as e:
            ephyr_logger().debug(str(e))

        self._refresh_pairs_list()
        self._update_banner()
        self._on_time_changed()
        self._persist()

    def closeEvent(self, event) -> None:  # noqa: N802
        try:
            self._session_manager.start_point_changed.disconnect(self._on_time_changed)
            self._session_manager.duration_ms_changed.disconnect(self._on_time_changed)
            self._session_manager.current_sweep_idx_changed.disconnect(self._on_sweep_changed)
        except Exception as e:
            ephyr_logger().debug(str(e))
        self._persist()
        try:
            self._frame_source.close()
        except Exception as e:
            ephyr_logger().debug(str(e))
        super().closeEvent(event)

    def _current_train(self) -> Optional[TtlTrain]:
        tid = self._train_combo.currentData()
        if tid is None:
            return None
        for train in self._trains:
            if int(train.train_id) == int(tid):
                return train
        return None

    def _signal_center_ms(self) -> float:
        header = self._session_manager.header
        sample_rate = float(header.sample_rate)
        gui = self._session_manager.gui_setup
        start_ms = (float(gui.start_point) / sample_rate) * 1000.0
        return start_ms + float(gui.duration_ms) / 2.0

    def _center_signal_on_ms(self, time_ms: float) -> None:
        try:
            header = self._session_manager.header
            sample_rate = float(header.sample_rate)
            duration_ms = float(self._session_manager.gui_setup.duration_ms)
            sweep_idx = int(self._session_manager.gui_setup.current_sweep_idx)
            sweep_points = int(header.number_of_points_per_sweep[sweep_idx])
            target_samples = int((float(time_ms) / 1000.0) * sample_rate)
            half_window = int((duration_ms / 2000.0) * sample_rate)
            new_start = max(0, target_samples - half_window)
            max_start = max(0, sweep_points - int((duration_ms / 1000.0) * sample_rate))
            new_start = min(new_start, max_start)
            self._session_manager.set_start_point(int(new_start))
        except Exception as e:
            ephyr_logger().debug(str(e))

    def _on_train_changed(self, _index: int = 0) -> None:
        if self._updating_ui or self._is_nwb:
            return
        train = self._current_train()
        if train is None:
            return
        # Prefer candidate pairs for this train
        for cand in self._candidates:
            if int(cand.train_id) == int(train.train_id) and cand.pairs:
                self._mapping.set_pairs(cand.pairs)
                break
        else:
            from ephyr_add_ons.labeling_utils.video_sync.detection import align_sequences

            self._mapping.set_pairs(
                align_sequences(train.times_ms.tolist(), self._flash_times_ms.tolist())
            )
        self._refresh_pairs_list()
        self._update_banner()
        self._on_time_changed()
        self._persist()

    def _refresh_pairs_list(self) -> None:
        self._updating_ui = True
        self._pairs_list.clear()
        target = "imaging" if self._is_nwb else "video"
        for i, (sig, vid) in enumerate(self._mapping.pairs):
            item = QListWidgetItem(f"#{i}: signal {sig:.2f} ms → {target} {vid:.2f} ms")
            item.setData(Qt.ItemDataRole.UserRole, i)
            self._pairs_list.addItem(item)
        self._updating_ui = False

    def _on_unlock_toggled(self, unlocked: bool) -> None:
        self._follow_signal = not unlocked
        self._unfollow_btn.setText("Lock to signal" if unlocked else "Unlock frame scrub")
        if self._follow_signal:
            self._on_time_changed()

    def _on_frame_spin_changed(self, frame_index: int) -> None:
        if self._updating_ui:
            return
        self._follow_signal = False
        self._unfollow_btn.blockSignals(True)
        self._unfollow_btn.setChecked(True)
        self._unfollow_btn.setText("Lock to signal")
        self._unfollow_btn.blockSignals(False)
        self._show_frame(int(frame_index))
        if self._meta.fps > 0:
            self._updating_ui = True
            self._video_ms_spin.setValue((int(frame_index) / self._meta.fps) * 1000.0)
            self._updating_ui = False

    def _on_video_ms_spin_changed(self, video_ms: float) -> None:
        if self._updating_ui:
            return
        if self._meta.fps <= 0:
            return
        frame_index = int(round((float(video_ms) / 1000.0) * self._meta.fps))
        frame_index = max(0, min(frame_index, max(0, self._meta.frame_count - 1)))
        self._updating_ui = True
        self._frame_spin.setValue(frame_index)
        self._updating_ui = False
        self._on_frame_spin_changed(frame_index)

    def _add_pair_from_current(self) -> None:
        signal_ms = self._signal_center_ms()
        if self._last_frame_index is not None and self._meta.fps > 0:
            video_ms = (self._last_frame_index / self._meta.fps) * 1000.0
        else:
            video_ms = float(self._video_ms_spin.value())
        self._mapping.add_pair(signal_ms, video_ms)
        self._follow_signal = True
        self._unfollow_btn.blockSignals(True)
        self._unfollow_btn.setChecked(False)
        self._unfollow_btn.setText("Unlock frame scrub")
        self._unfollow_btn.blockSignals(False)
        self._refresh_pairs_list()
        self._update_banner()
        self._on_time_changed()
        self._persist()

    def _remove_selected_pair(self) -> None:
        item = self._pairs_list.currentItem()
        if item is None:
            return
        idx = item.data(Qt.ItemDataRole.UserRole)
        if idx is None:
            return
        self._mapping.remove_pair_at(int(idx))
        self._refresh_pairs_list()
        self._update_banner()
        self._on_time_changed()
        self._persist()

    def _step_ttl(self, delta: int) -> None:
        train = self._current_train()
        if train is None or train.times_ms.size == 0:
            return
        times = np.asarray(train.times_ms, dtype=np.float64)
        center = self._signal_center_ms()
        if delta > 0:
            nxt = times[times > center + 1e-6]
            if nxt.size == 0:
                return
            target = float(nxt[0])
        else:
            prv = times[times < center - 1e-6]
            if prv.size == 0:
                return
            target = float(prv[-1])
        self._center_signal_on_ms(target)

    def _step_flash(self, delta: int) -> None:
        if self._flash_times_ms.size == 0:
            return
        # Current video time from mapping
        center = self._signal_center_ms()
        video_ms = self._mapping.to_video_ms(center) if self._mapping.pairs else center
        times = self._flash_times_ms
        if delta > 0:
            nxt = times[times > video_ms + 1e-6]
            if nxt.size == 0:
                return
            target_video = float(nxt[0])
        else:
            prv = times[times < video_ms - 1e-6]
            if prv.size == 0:
                return
            target_video = float(prv[-1])
        if self._mapping.pairs:
            target_signal = self._mapping.to_signal_ms(target_video)
        else:
            target_signal = target_video
        self._center_signal_on_ms(target_signal)

    def _update_banner(self) -> None:
        interval = self._mapping.trusted_interval_signal_ms
        if interval is None:
            if self._is_nwb:
                self._banner.setText(
                    "No timestamp map yet. Re-run NWB imaging sync or add pairs manually."
                )
            else:
                self._banner.setText(
                    "No sync pairs yet. Add pairs or select a TTL sequence to build a time map."
                )
            return
        lo, hi = interval
        media = "imaging" if self._is_nwb else "video"
        bits = [
            f"Trusted sync on signal [{lo:.1f}, {hi:.1f}] ms "
            f"(a={self._mapping.a:.6f}, b={self._mapping.b:.2f} ms)."
        ]
        if self._is_nwb:
            bits.insert(
                0,
                "NWB imaging aligned by timestamps (shared session clock via starting_time).",
            )
        if self._signal_duration_ms > 0 and (lo > 0 or hi < self._signal_duration_ms):
            bits.append(
                "Markers do not cover the full signal duration — outside this interval "
                "the map is extrapolated."
            )
        if self._meta.duration_ms > 0:
            v0 = self._mapping.to_video_ms(lo)
            v1 = self._mapping.to_video_ms(hi)
            if v0 > 50 or v1 < self._meta.duration_ms - 50:
                bits.append(
                    f"{media.capitalize()} extends beyond the trusted marker span; "
                    "edge frames may be approximate."
                )
        self._banner.setText(" ".join(bits))

    def _on_sweep_changed(self, *_args) -> None:
        self._on_time_changed()

    def _on_time_changed(self, *_args) -> None:
        if not self._follow_signal:
            return
        try:
            center_ms = self._signal_center_ms()
            media = "imaging" if self._is_nwb else "video"
            if not self._mapping.pairs:
                self._status.setText(
                    f"Signal center {center_ms:.1f} ms — no mapping; showing first frame."
                )
                self._show_frame(0)
                return

            video_ms = self._mapping.to_video_ms(center_ms)
            frame_idx = self._mapping.to_frame_index(center_ms, self._meta.fps)
            trusted = self._mapping.is_trusted(center_ms)
            in_range = 0 <= frame_idx < max(1, self._meta.frame_count)

            if not in_range:
                clamped = int(np.clip(frame_idx, 0, max(0, self._meta.frame_count - 1)))
                trust_txt = "trusted" if trusted else "EXTRAPOLATED"
                self._status.setText(
                    f"Signal {center_ms:.1f} ms → {media} {video_ms:.1f} ms "
                    f"(frame {frame_idx}) OUT OF RANGE — showing edge frame {clamped} [{trust_txt}]"
                )
                self._show_frame(clamped)
                return

            trust_txt = "trusted" if trusted else "EXTRAPOLATED"
            self._status.setText(
                f"Signal {center_ms:.1f} ms → {media} {video_ms:.1f} ms → frame {frame_idx} [{trust_txt}]"
            )
            self._show_frame(frame_idx)
        except Exception as e:
            ephyr_logger().debug(str(e))

    def _show_frame(self, frame_index: int) -> None:
        frame_index = int(frame_index)
        if self._last_frame_index == frame_index:
            self._sync_scrub_spins(frame_index)
            return
        try:
            frame = self._frame_source.read_bgr(frame_index)
        except Exception:
            frame = read_frame_bgr(self._meta.path, frame_index) if not self._is_nwb else None
        pix = _bgr_to_qpixmap(frame)
        if pix is None:
            self._frame_label.setText("Cannot read frame")
            self._last_frame_index = None
            return
        self._frame_label.setPixmap(pix)
        self._last_frame_index = frame_index
        self._sync_scrub_spins(frame_index)

    def _sync_scrub_spins(self, frame_index: int) -> None:
        self._updating_ui = True
        self._frame_spin.setValue(int(frame_index))
        if self._meta.fps > 0:
            self._video_ms_spin.setValue((int(frame_index) / self._meta.fps) * 1000.0)
        self._updating_ui = False

    def _persist(self) -> None:
        try:
            self._mapping.save(mapping_path(self._add_on_data_dir))
        except Exception as e:
            ephyr_logger().debug(str(e))
