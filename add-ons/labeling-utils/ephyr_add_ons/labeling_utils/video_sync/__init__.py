"""Video sync add-on.

Runnable. Two sources:

1. Video file — calibrate against the signal timeline using TTL pulses (AUX)
   matched to brightness flashes, then open a window whose frame follows the
   center of the visible signal window.
2. NWB imaging by timestamps — show ``OnePhotonSeries`` / ``TwoPhotonSeries`` /
   ``ImageSeries`` frames aligned via rate / starting_time / timestamps
   (no flash detection).
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
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

from ephyr.core.add_ons.base import BaseAddOn
from ephyr.logger import ephyr_logger

from ephyr_add_ons.labeling_utils._common import LabelingUtilsBase
from ephyr_add_ons.labeling_utils.video_sync.detection import (
    EDGE_FALLING,
    EDGE_RISING,
    SOURCE_NWB,
    SOURCE_VIDEO,
    VideoFileFrameSource,
    cluster_ttl_trains,
    detect_flashes_in_video,
    detect_ttl_edges,
    match_trains_to_flashes,
    open_video_meta,
    samples_to_ms,
)
from ephyr_add_ons.labeling_utils.video_sync.mapping import TimeMapping, mapping_path
from ephyr_add_ons.labeling_utils.video_sync.window import VideoSyncWindow

MODE_AUTO = "auto"
MODE_MANUAL = "manual"

VIDEO_FILTERS = "Video files (*.mp4 *.avi *.mkv *.mov *.MP4 *.AVI *.MKV *.MOV);;All files (*)"
NWB_FILTERS = "NWB files (*.nwb);;All files (*)"


class VideoSyncAddOn(LabelingUtilsBase, BaseAddOn):
    TRANSFORMATION = False
    VIEWABLE = False
    RUNNABLE = True

    def __init__(self):
        self._window: Optional[VideoSyncWindow] = None
        self._settings_window: Optional[QWidget] = None

    def _guess_nwb_path(self, session_manager, header) -> str:
        try:
            from ephyr.converter.channel_order import resolve_source_path

            folder = getattr(session_manager, "ephyr_experiment_folder", None)
            path = resolve_source_path(folder, header)
            if path is not None and path.is_file() and path.suffix.lower() == ".nwb":
                return str(path)
        except Exception as e:
            ephyr_logger().debug(str(e))
        return ""

    def _ask_parameters(self, session_manager, header, add_on_data_dir: Path) -> Optional[QWidget]:
        params = self.load_params(add_on_data_dir)
        common = self.load_common(add_on_data_dir)
        groups = self.channel_groups(session_manager)

        dialog = QWidget()
        dialog.setWindowTitle("Video sync")
        dialog.setWindowFlags(Qt.WindowType.Window)
        dialog.setMinimumWidth(560)
        dialog.setMinimumHeight(640)
        outer = QVBoxLayout(dialog)
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_widget = QWidget()
        root = QVBoxLayout(scroll_widget)

        source_combo = QComboBox()
        source_combo.addItem("Video file (TTL ↔ flashes)", SOURCE_VIDEO)
        source_combo.addItem("NWB imaging by timestamps", SOURCE_NWB)
        source_idx = source_combo.findData(str(params.get("source_kind", SOURCE_VIDEO)))
        source_combo.setCurrentIndex(max(0, source_idx))
        source_row = QHBoxLayout()
        source_row.addWidget(QLabel("Source:"))
        source_row.addWidget(source_combo, stretch=1)
        root.addLayout(source_row)

        # --- Video mode panel ---
        video_panel = QWidget()
        video_form = QFormLayout(video_panel)

        video_row = QHBoxLayout()
        video_edit = QLineEdit(str(params.get("video_path", "")))
        video_edit.setPlaceholderText("Select a video file…")
        browse_video_btn = QPushButton("Browse…")

        def _browse_video() -> None:
            start = video_edit.text().strip() or str(Path.home())
            path, _ = QFileDialog.getOpenFileName(dialog, "Select video", start, VIDEO_FILTERS)
            if path:
                video_edit.setText(path)

        browse_video_btn.clicked.connect(_browse_video)
        video_row.addWidget(video_edit, stretch=1)
        video_row.addWidget(browse_video_btn)
        video_form.addRow("Video file:", video_row)

        group_combo = None
        channels_list = None
        if groups:
            group_combo, channels_list = self.build_group_channel_selector(
                video_form,
                groups,
                header,
                preferred_group_idx=int(common.get("group_idx", 0)),
                preferred_channels=common.get("channel_indexes", []),
            )
        else:
            video_form.addRow(
                QLabel("No channel groups configured — required for TTL ↔ flash mode.")
            )

        edge_combo = QComboBox()
        edge_combo.addItem("Rising edge", EDGE_RISING)
        edge_combo.addItem("Falling edge", EDGE_FALLING)
        edge_idx = edge_combo.findData(str(params.get("edge", EDGE_RISING)))
        edge_combo.setCurrentIndex(max(0, edge_idx))
        video_form.addRow("TTL edge:", edge_combo)

        ttl_threshold_spin = QDoubleSpinBox()
        ttl_threshold_spin.setDecimals(6)
        ttl_threshold_spin.setRange(-1e9, 1e9)
        ttl_threshold_spin.setValue(float(params.get("ttl_threshold", 0.5)))
        video_form.addRow("TTL threshold:", ttl_threshold_spin)

        ttl_distance_spin = QDoubleSpinBox()
        ttl_distance_spin.setDecimals(3)
        ttl_distance_spin.setRange(0.0, 1e9)
        ttl_distance_spin.setValue(float(params.get("ttl_distance_ms", 1.0)))
        video_form.addRow("TTL min distance (ms):", ttl_distance_spin)

        flash_sens_spin = QDoubleSpinBox()
        flash_sens_spin.setDecimals(2)
        flash_sens_spin.setRange(0.1, 50.0)
        flash_sens_spin.setSingleStep(0.25)
        flash_sens_spin.setValue(float(params.get("flash_sensitivity", 2.5)))
        video_form.addRow("Flash sensitivity:", flash_sens_spin)

        flash_distance_spin = QDoubleSpinBox()
        flash_distance_spin.setDecimals(3)
        flash_distance_spin.setRange(0.0, 1e9)
        flash_distance_spin.setValue(float(params.get("flash_distance_ms", 50.0)))
        video_form.addRow("Flash min distance (ms):", flash_distance_spin)

        sample_fps_spin = QDoubleSpinBox()
        sample_fps_spin.setDecimals(1)
        sample_fps_spin.setRange(0.0, 240.0)
        sample_fps_spin.setSpecialValueText("auto")
        sample_fps_spin.setValue(float(params.get("flash_sample_fps", 0.0)))
        video_form.addRow("Flash sample fps (0=auto):", sample_fps_spin)

        mode_combo = QComboBox()
        mode_combo.addItem("Auto match", MODE_AUTO)
        mode_combo.addItem("Manual", MODE_MANUAL)
        mode_idx = mode_combo.findData(str(params.get("mode", MODE_AUTO)))
        mode_combo.setCurrentIndex(max(0, mode_idx))
        video_form.addRow("Mode:", mode_combo)

        # --- NWB mode panel ---
        nwb_panel = QWidget()
        nwb_form = QFormLayout(nwb_panel)

        default_nwb = str(params.get("nwb_path", "")).strip() or self._guess_nwb_path(
            session_manager, header
        )
        nwb_row = QHBoxLayout()
        nwb_edit = QLineEdit(default_nwb)
        nwb_edit.setPlaceholderText("Select an NWB file with imaging…")
        browse_nwb_btn = QPushButton("Browse…")
        refresh_series_btn = QPushButton("Load series")
        nwb_row.addWidget(nwb_edit, stretch=1)
        nwb_row.addWidget(browse_nwb_btn)
        nwb_row.addWidget(refresh_series_btn)
        nwb_form.addRow("NWB file:", nwb_row)

        series_combo = QComboBox()
        series_combo.setMinimumWidth(360)
        nwb_form.addRow("Imaging series:", series_combo)
        nwb_form.addRow(
            QLabel(
                "Frames follow the signal window via rate / starting_time / timestamps "
                "(no TTL or LED flash detection)."
            )
        )

        def _populate_series(prefer: str = "") -> None:
            series_combo.clear()
            path_text = nwb_edit.text().strip()
            if not path_text or not Path(path_text).is_file():
                series_combo.addItem("(select a valid NWB file)", "")
                return
            try:
                from ephyr_add_ons.labeling_utils.video_sync.nwb_imaging import list_imaging_series

                series = list_imaging_series(Path(path_text))
            except Exception as e:
                series_combo.addItem(f"(failed to list series: {e})", "")
                return
            if not series:
                series_combo.addItem("(no ImageSeries / OnePhoton / TwoPhoton found)", "")
                return
            prefer = prefer or str(params.get("series_location", ""))
            pick = 0
            for i, info in enumerate(series):
                label = (
                    f"{info.name} — {info.location} — "
                    f"{info.n_frames} frames @ {info.fps:.2f} Hz"
                )
                series_combo.addItem(label, info.location)
                if prefer and info.location == prefer:
                    pick = i
            series_combo.setCurrentIndex(pick)

        def _browse_nwb() -> None:
            start = nwb_edit.text().strip() or str(Path.home())
            path, _ = QFileDialog.getOpenFileName(dialog, "Select NWB", start, NWB_FILTERS)
            if path:
                nwb_edit.setText(path)
                _populate_series()

        browse_nwb_btn.clicked.connect(_browse_nwb)
        refresh_series_btn.clicked.connect(lambda: _populate_series())

        root.addWidget(video_panel)
        root.addWidget(nwb_panel)
        root.addStretch(1)

        def _on_source_changed(_index: int = 0) -> None:
            is_video = source_combo.currentData() == SOURCE_VIDEO
            video_panel.setVisible(is_video)
            nwb_panel.setVisible(not is_video)
            if not is_video and series_combo.count() == 0:
                _populate_series()

        source_combo.currentIndexChanged.connect(_on_source_changed)
        _on_source_changed()
        if source_combo.currentData() == SOURCE_NWB and default_nwb:
            _populate_series()

        scroll_area.setWidget(scroll_widget)
        outer.addWidget(scroll_area)
        status = QLabel("")
        status.setWordWrap(True)
        outer.addWidget(status)
        progress = QProgressBar()
        progress.setRange(0, 100)
        progress.setValue(0)
        outer.addWidget(progress)
        actions = QHBoxLayout()
        btn_close = QPushButton("Close")
        btn_run = QPushButton("Run")
        actions.addStretch(1)
        actions.addWidget(btn_close)
        actions.addWidget(btn_run)
        outer.addLayout(actions)
        btn_close.clicked.connect(dialog.close)

        def _collect_params() -> Optional[dict]:
            source_kind = str(source_combo.currentData())
            if source_kind == SOURCE_NWB:
                nwb_path = nwb_edit.text().strip()
                series_location = str(series_combo.currentData() or "")
                if not nwb_path or not Path(nwb_path).is_file():
                    QMessageBox.warning(dialog, "Video sync", "Select a valid NWB file.")
                    return None
                if not series_location:
                    QMessageBox.warning(dialog, "Video sync", "Select an imaging series.")
                    return None
                payload = {
                    "source_kind": SOURCE_NWB,
                    "nwb_path": nwb_path,
                    "series_location": series_location,
                    "video_path": video_edit.text().strip() or str(params.get("video_path", "")),
                    "edge": str(params.get("edge", EDGE_RISING)),
                    "ttl_threshold": float(params.get("ttl_threshold", 0.5)),
                    "ttl_distance_ms": float(params.get("ttl_distance_ms", 1.0)),
                    "flash_sensitivity": float(params.get("flash_sensitivity", 2.5)),
                    "flash_distance_ms": float(params.get("flash_distance_ms", 50.0)),
                    "flash_sample_fps": float(params.get("flash_sample_fps", 0.0)),
                    "mode": str(params.get("mode", MODE_AUTO)),
                }
                self.save_params(add_on_data_dir, payload)
                return {
                    "source_kind": SOURCE_NWB,
                    "nwb_path": nwb_path,
                    "series_location": series_location,
                }

            if not groups or group_combo is None or channels_list is None:
                QMessageBox.warning(dialog, "Video sync", "Channel groups are required for video mode.")
                return None

            video_path = video_edit.text().strip()
            if not video_path or not Path(video_path).is_file():
                QMessageBox.warning(dialog, "Video sync", "Select a valid video file.")
                return None

            channels = self.selected_channels(channels_list)
            if not channels:
                QMessageBox.warning(dialog, "Video sync", "Select a TTL channel.")
                return None
            channel = int(channels[0])

            try:
                open_video_meta(video_path)
            except Exception as e:
                QMessageBox.warning(dialog, "Video sync", f"Cannot open video:\n{e}")
                return None

            self.save_common(
                add_on_data_dir,
                {"group_idx": int(group_combo.currentData()), "channel_indexes": [channel]},
            )
            payload = {
                "source_kind": SOURCE_VIDEO,
                "video_path": video_path,
                "edge": str(edge_combo.currentData()),
                "ttl_threshold": float(ttl_threshold_spin.value()),
                "ttl_distance_ms": float(ttl_distance_spin.value()),
                "flash_sensitivity": float(flash_sens_spin.value()),
                "flash_distance_ms": float(flash_distance_spin.value()),
                "flash_sample_fps": float(sample_fps_spin.value()),
                "mode": str(mode_combo.currentData()),
                "nwb_path": nwb_edit.text().strip() or str(params.get("nwb_path", default_nwb)),
                "series_location": str(series_combo.currentData() or params.get("series_location", "")),
            }
            self.save_params(add_on_data_dir, payload)
            return {"channel": channel, **payload}

        def _on_run() -> None:
            collected = _collect_params()
            if collected is None:
                return
            btn_run.setEnabled(False)
            try:
                self._start_sync(session_manager, add_on_data_dir, collected, status, progress)
            finally:
                btn_run.setEnabled(True)

        btn_run.clicked.connect(_on_run)
        return dialog

    def _open_window(
        self,
        session_manager,
        add_on_data_dir: Path,
        meta,
        mapping: TimeMapping,
        signal_duration_ms: float,
        trains=None,
        flash_times_ms=None,
        candidates=None,
        selected_train_id=None,
        frame_source=None,
        status_message: str = "",
    ):
        try:
            if self._window is not None:
                self._window.close()
        except Exception as e:
            ephyr_logger().debug(str(e))

        self._window = VideoSyncWindow(
            session_manager=session_manager,
            add_on_data_dir=add_on_data_dir,
            video_meta=meta,
            trains=trains or [],
            flash_times_ms=flash_times_ms or [],
            candidates=candidates or [],
            initial_mapping=mapping,
            selected_train_id=selected_train_id,
            signal_duration_ms=signal_duration_ms,
            frame_source=frame_source,
        )
        self._window.show()
        self._window.raise_()
        self._window.activateWindow()
        return status_message

    def _run_nwb(self, session_manager, add_on_data_dir: Path, params: dict, signal_duration_ms: float):
        try:
            from ephyr_add_ons.labeling_utils.video_sync.nwb_imaging import (
                NwbFrameSource,
                build_timestamp_mapping,
                electrical_starting_time_s,
                ephys_location_from_header,
                list_imaging_series,
                meta_from_imaging_info,
            )
        except ImportError:
            QMessageBox.warning(
                None,
                "Video sync",
                "pynwb is required for NWB imaging mode. Install it in the Ephyr environment:\n"
                "  pip install pynwb",
            )
            return

        nwb_path = Path(params["nwb_path"])
        series_location = str(params["series_location"])
        yield {"progress": 10, "message": "Opening NWB imaging series…"}
        try:
            series_list = list_imaging_series(nwb_path)
            info = next((s for s in series_list if s.location == series_location), None)
            if info is None:
                raise ValueError(f"Imaging series not found: {series_location}")
            meta = meta_from_imaging_info(nwb_path, info)
            preferred = ephys_location_from_header(session_manager.header)
            ephys_start = electrical_starting_time_s(nwb_path, preferred_location=preferred)
            if ephys_start is None:
                ephys_start = 0.0
            mapping = build_timestamp_mapping(
                imaging_start_s=info.starting_time_s,
                ephys_start_s=float(ephys_start),
                signal_duration_ms=signal_duration_ms,
                imaging_duration_ms=info.duration_ms,
            )
            frame_source = NwbFrameSource(nwb_path, series_location)
        except Exception as e:
            ephyr_logger().debug(str(e))
            QMessageBox.warning(None, "Video sync", f"NWB imaging sync failed:\n{e}")
            return

        yield {"progress": 90, "message": "Opening imaging sync window…"}
        msg = self._open_window(
            session_manager=session_manager,
            add_on_data_dir=add_on_data_dir,
            meta=meta,
            mapping=mapping,
            signal_duration_ms=signal_duration_ms,
            frame_source=frame_source,
            status_message=(
                f"NWB imaging sync ready: {info.location}, "
                f"{info.n_frames} frames @ {info.fps:.2f} Hz, "
                f"Δt={mapping.b:.1f} ms"
            ),
        )
        yield {"progress": 100, "message": msg}

    def _run_video(self, session_manager, add_on_data_dir: Path, params: dict, signal_duration_ms: float):
        try:
            import cv2  # noqa: F401
        except ImportError:
            QMessageBox.warning(
                None,
                "Video sync",
                "opencv-python is required. Install it in the Ephyr environment:\n"
                "  pip install opencv-python>=4.8",
            )
            return

        header = session_manager.header
        sample_rate = float(header.sample_rate)
        sweep_idx = int(session_manager.gui_setup.current_sweep_idx)
        sweep_points = int(header.number_of_points_per_sweep[sweep_idx])
        channel = int(params["channel"])
        video_path = str(params["video_path"])
        mode = str(params.get("mode", MODE_AUTO))

        yield {"progress": 5, "message": "Loading video metadata…"}
        try:
            meta = open_video_meta(video_path)
        except Exception as e:
            QMessageBox.warning(None, "Video sync", f"Cannot open video:\n{e}")
            return

        yield {"progress": 15, "message": "Detecting TTL pulses…"}
        distance_samples = (
            max(1, int((params["ttl_distance_ms"] * sample_rate) / 1000.0))
            if params["ttl_distance_ms"] > 0
            else 1
        )
        try:
            matrix = self.channel_matrix_from_session(
                session_manager, [channel], sweep_idx, 0, sweep_points, sample_rate
            )
            signal = np.asarray(matrix[0], dtype=np.float64)
            peak_samples = detect_ttl_edges(
                signal,
                threshold=float(params["ttl_threshold"]),
                edge=str(params["edge"]),
                min_distance_samples=distance_samples,
            )
            ttl_times_ms = samples_to_ms(peak_samples, sample_rate)
        except Exception as e:
            ephyr_logger().debug(str(e))
            QMessageBox.warning(None, "Video sync", f"TTL detection failed:\n{e}")
            return

        yield {
            "progress": 35,
            "message": f"Found {ttl_times_ms.size} TTL pulse(s); scanning video for flashes…",
        }

        flash_progress = {"last": 35}

        def _flash_progress(pct: int) -> None:
            flash_progress["last"] = 35 + int(0.45 * pct)

        try:
            flash_times_ms, meta = detect_flashes_in_video(
                video_path,
                sensitivity=float(params["flash_sensitivity"]),
                min_distance_ms=float(params["flash_distance_ms"]),
                sample_fps=float(params.get("flash_sample_fps", 0.0) or 0.0),
                progress_callback=_flash_progress,
            )
        except Exception as e:
            ephyr_logger().debug(str(e))
            QMessageBox.warning(None, "Video sync", f"Flash detection failed:\n{e}")
            return

        yield {
            "progress": 80,
            "message": f"Found {flash_times_ms.size} flash(es); clustering TTL trains…",
        }

        trains = cluster_ttl_trains(ttl_times_ms.tolist())
        candidates: List = []
        selected_train_id = None
        mapping = TimeMapping()

        if mode == MODE_AUTO and trains and flash_times_ms.size:
            candidates = match_trains_to_flashes(trains, flash_times_ms.tolist())
            if candidates:
                selected_train_id = int(candidates[0].train_id)
                mapping.set_pairs(candidates[0].pairs)
        elif trains and flash_times_ms.size:
            candidates = match_trains_to_flashes(trains, flash_times_ms.tolist(), min_score=0.0)

        saved = TimeMapping.load(mapping_path(add_on_data_dir))
        if saved is not None and saved.pairs and mode == MODE_MANUAL:
            mapping = saved

        if not ttl_times_ms.size and not flash_times_ms.size:
            QMessageBox.information(
                None,
                "Video sync",
                "No TTL pulses and no flashes were detected. "
                "You can still open the window and add pairs manually.",
            )
        elif not ttl_times_ms.size:
            QMessageBox.information(
                None,
                "Video sync",
                "No TTL pulses detected on the selected channel. "
                "Adjust threshold/edge or add pairs manually.",
            )
        elif not flash_times_ms.size:
            QMessageBox.information(
                None,
                "Video sync",
                "No flashes detected in the video. "
                "Adjust flash sensitivity or add pairs manually.",
            )

        yield {"progress": 95, "message": "Opening sync window…"}
        msg = self._open_window(
            session_manager=session_manager,
            add_on_data_dir=add_on_data_dir,
            meta=meta,
            mapping=mapping,
            signal_duration_ms=signal_duration_ms,
            trains=trains,
            flash_times_ms=flash_times_ms.tolist(),
            candidates=candidates,
            selected_train_id=selected_train_id,
            frame_source=VideoFileFrameSource(video_path),
            status_message=(
                f"Video sync ready: {ttl_times_ms.size} TTL, {flash_times_ms.size} flashes, "
                f"{len(trains)} train(s), {len(mapping.pairs)} pair(s)"
            ),
        )
        yield {"progress": 100, "message": msg}

    def _start_sync(self, session_manager, add_on_data_dir: Path, params: dict, status, progress):
        header = session_manager.header
        sweep_idx = int(session_manager.gui_setup.current_sweep_idx)
        sample_rate = float(header.sample_rate)
        sweep_points = int(header.number_of_points_per_sweep[sweep_idx])
        signal_duration_ms = (sweep_points / sample_rate) * 1000.0 if sample_rate > 0 else 0.0
        source_kind = str(params.get("source_kind", SOURCE_VIDEO))
        worker = (
            self._run_nwb(session_manager, add_on_data_dir, params, signal_duration_ms)
            if source_kind == SOURCE_NWB
            else self._run_video(session_manager, add_on_data_dir, params, signal_duration_ms)
        )
        if worker is None:
            return
        for yielded in worker:
            if isinstance(yielded, dict):
                message = str(yielded.get("message", ""))
                value = yielded.get("progress")
                if message:
                    status.setText(message)
                if value is not None:
                    progress.setValue(max(0, min(100, int(value))))
                QApplication.processEvents()

    def run(self, session_manager, add_on_data_dir):
        header = session_manager.header
        add_on_data_dir = Path(add_on_data_dir)
        try:
            if self._settings_window is not None:
                self._settings_window.close()
        except Exception as e:
            ephyr_logger().debug(str(e))
        self._settings_window = self._ask_parameters(session_manager, header, add_on_data_dir)
        if self._settings_window is None:
            return False
        self._settings_window.show()
        self._settings_window.raise_()
        self._settings_window.activateWindow()
        return False


__all__ = ["VideoSyncAddOn"]
