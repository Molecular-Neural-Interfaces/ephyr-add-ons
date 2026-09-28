"""Calibrated, screen-fixed EEG paper grid."""

from __future__ import annotations

from pathlib import Path
from collections import defaultdict
from typing import Any, Iterable, List, Optional, Sequence, Tuple

from PyQt6.QtCore import QRect, Qt
from PyQt6.QtGui import QColor, QPainter, QPen
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QFormLayout,
    QHBoxLayout,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ephyr.core.add_ons.base import BaseAddOn, ViewEntitiesZIndexEnum
from ephyr.core.add_ons.common.mixin import EphyrAddOnMixin
from ephyr.logger import ephyr_logger


SPEEDS_MM_PER_S: Tuple[int, ...] = (15, 30, 60)
SENSITIVITIES_UV_PER_MM: Tuple[int, ...] = (3, 5, 7, 10, 15, 20)
MINOR_INTERVAL_MS = 200.0
MAJOR_INTERVAL_MS = 1000.0
MM_PER_INCH = 25.4
FALLBACK_DPI = 96.0


def pixels_per_mm(widget: QWidget) -> float:
    """Return physical, device-independent pixels per millimetre for a widget."""
    screen = widget.screen()
    dpi = float(screen.physicalDotsPerInchX()) if screen is not None else 0.0
    if dpi <= 0.0 and screen is not None:
        dpi = float(screen.logicalDotsPerInchX())
    if dpi <= 0.0:
        dpi = FALLBACK_DPI
    return dpi / MM_PER_INCH


def duration_for_speed(width_px: float, px_per_mm: float, speed_mm_per_s: float) -> int:
    """Duration in milliseconds needed to show ``speed_mm_per_s``."""
    if width_px <= 0.0 or px_per_mm <= 0.0 or speed_mm_per_s <= 0.0:
        raise ValueError("Width, pixel density, and speed must be positive.")
    width_mm = float(width_px) / float(px_per_mm)
    return max(1, int(round((width_mm / float(speed_mm_per_s)) * 1000.0)))


def voltage_scale_for_sensitivity(
    cell_height_px: float,
    px_per_mm: float,
    sensitivity_uv_per_mm: float,
) -> float:
    """Full-cell voltage scale corresponding to an EEG sensitivity."""
    if cell_height_px <= 0.0 or px_per_mm <= 0.0 or sensitivity_uv_per_mm <= 0.0:
        raise ValueError("Cell height, pixel density, and sensitivity must be positive.")
    return (float(cell_height_px) / float(px_per_mm)) * float(sensitivity_uv_per_mm)


def grid_x_positions(rect: QRect, duration_ms: float) -> Tuple[List[int], List[int]]:
    """Return minor and major line positions relative to a screen-fixed cell."""
    if duration_ms <= 0.0 or rect.width() <= 0:
        return [], []
    span = max(0, rect.width() - 1)
    steps = int(float(duration_ms) // MINOR_INTERVAL_MS)
    minor: List[int] = []
    major: List[int] = []
    for step in range(steps + 1):
        elapsed_ms = step * MINOR_INTERVAL_MS
        x = rect.left() + int(round((elapsed_ms / float(duration_ms)) * span))
        if step % int(MAJOR_INTERVAL_MS / MINOR_INTERVAL_MS) == 0:
            major.append(x)
        else:
            minor.append(x)
    return minor, major


def visible_signal_widget() -> Optional[QWidget]:
    app = QApplication.instance()
    if app is None:
        return None
    candidates = [
        widget
        for widget in QApplication.allWidgets()
        if widget.isVisible()
        and hasattr(widget, "_digital_channel_rects")
        and hasattr(widget, "_group_layouts")
        and hasattr(widget, "_axis_width")
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda widget: widget.width() * widget.height())


def group_cell_size(signal_widget: QWidget, group: Any) -> Optional[Tuple[float, float]]:
    channels = {int(ch) for ch in (getattr(group, "channel_indexes", []) or [])}
    rects = [
        rect
        for channel_idx, rect in getattr(signal_widget, "_digital_channel_rects", [])
        if int(channel_idx) in channels and rect.width() > 0 and rect.height() > 0
    ]
    if not rects:
        return None

    def median(values: Iterable[int]) -> float:
        ordered = sorted(values)
        middle = len(ordered) // 2
        if len(ordered) % 2:
            return float(ordered[middle])
        return (float(ordered[middle - 1]) + float(ordered[middle])) / 2.0

    return median(rect.width() for rect in rects), median(rect.height() for rect in rects)


class _EEGGridWindow(QWidget):
    def __init__(self, add_on: "EEGGridAddOn", session_manager, add_on_data_dir: Path, groups: Sequence[Any]):
        super().__init__()
        self._add_on = add_on
        self._session_manager = session_manager
        self._add_on_data_dir = Path(add_on_data_dir)
        self._groups = list(groups)

        self.setWindowTitle("EEG grid")
        self.setWindowFlags(Qt.WindowType.Window)
        self.setMinimumWidth(420)

        params = add_on.load_params(self._add_on_data_dir)
        layout = QVBoxLayout(self)
        form = QFormLayout()
        self._groups_list = add_on.build_groups_multi_selector(
            form,
            self._groups,
            preferred_keys=params.get("target_group_keys", []),
            label="Channel groups:",
            non_aux_only=True,
        )

        self._speed_combo = QComboBox()
        for speed in SPEEDS_MM_PER_S:
            self._speed_combo.addItem(f"{speed} mm/s", speed)
        speed_idx = self._speed_combo.findData(int(params.get("speed_mm_per_s", 30)))
        self._speed_combo.setCurrentIndex(max(0, speed_idx))
        form.addRow("Sweep speed:", self._speed_combo)

        self._sensitivity_combo = QComboBox()
        for sensitivity in SENSITIVITIES_UV_PER_MM:
            self._sensitivity_combo.addItem(f"{sensitivity} µV/mm", sensitivity)
        sensitivity_idx = self._sensitivity_combo.findData(
            int(params.get("sensitivity_uv_per_mm", 10))
        )
        self._sensitivity_combo.setCurrentIndex(max(0, sensitivity_idx))
        form.addRow("Sensitivity:", self._sensitivity_combo)
        layout.addLayout(form)

        actions = QHBoxLayout()
        btn_close = QPushButton("Close")
        btn_apply = QPushButton("Apply")
        actions.addStretch(1)
        actions.addWidget(btn_close)
        actions.addWidget(btn_apply)
        layout.addLayout(actions)
        btn_close.clicked.connect(self.close)
        btn_apply.clicked.connect(self._apply)

    def _apply(self) -> None:
        target_keys = self._add_on.selected_group_keys(self._groups_list)
        if not target_keys:
            QMessageBox.warning(self, "EEG grid", "Select at least one channel group.")
            return
        self._add_on.apply_calibration(
            self._session_manager,
            self._add_on_data_dir,
            self._groups,
            target_keys,
            int(self._speed_combo.currentData()),
            int(self._sensitivity_combo.currentData()),
            parent=self,
        )


class EEGGridAddOn(EphyrAddOnMixin, BaseAddOn):
    """Configure clinical EEG scaling and draw its vertical timing grid."""

    TRANSFORMATION = False
    VIEWABLE = True
    RUNNABLE = True
    Z_INDEX = ViewEntitiesZIndexEnum.MIDDLE_LINE.value + 1

    def __init__(self):
        self._window: Optional[_EEGGridWindow] = None

    def apply_calibration(
        self,
        session_manager,
        add_on_data_dir: Path,
        groups: Sequence[Any],
        target_keys: List[str],
        speed: int,
        sensitivity: int,
        parent: Optional[QWidget] = None,
    ) -> None:
        gui_setup = session_manager.gui_setup
        if gui_setup is None:
            return
        add_on_data_dir = Path(add_on_data_dir)
        self.save_params(
            add_on_data_dir,
            {
                "target_group_keys": target_keys,
                "speed_mm_per_s": speed,
                "sensitivity_uv_per_mm": sensitivity,
            },
        )

        signal_widget = visible_signal_widget()
        if signal_widget is None:
            QMessageBox.warning(
                parent,
                "EEG grid",
                "The signal panel is not visible, so physical scaling cannot be calibrated.",
            )
            return

        px_per_mm = pixels_per_mm(signal_widget)
        selected_groups = self.resolve_groups_by_keys(
            groups,
            target_keys,
            non_aux_only=True,
            fallback_all=False,
        )
        missing_geometry: List[str] = []
        cell_widths: List[float] = []
        for group in selected_groups:
            cell_size = group_cell_size(signal_widget, group)
            if cell_size is None:
                missing_geometry.append(str(getattr(group, "name", "Unnamed")))
                continue
            cell_width, cell_height = cell_size
            cell_widths.append(cell_width)
            scale = voltage_scale_for_sensitivity(cell_height, px_per_mm, sensitivity)
            buckets = defaultdict(list)
            for channel_idx in getattr(group, "channel_indexes", []) or []:
                setup = gui_setup.channels_setup.get(int(channel_idx))
                buckets[(
                    float(getattr(setup, "y_offset", 0.0)),
                    str(getattr(setup, "color", "#000000")),
                    str(getattr(setup, "info", "")),
                )].append(int(channel_idx))
            for (y_offset, color, info), channel_indexes in buckets.items():
                session_manager.set_channels_setup(
                    channel_indexes,
                    scale=scale,
                    y_offset=y_offset,
                    color=color,
                    info=info,
                )

        if not cell_widths:
            QMessageBox.warning(
                parent,
                "EEG grid",
                "None of the selected groups has visible cells to calibrate.",
            )
            return

        ordered_widths = sorted(cell_widths)
        reference_width = ordered_widths[len(ordered_widths) // 2]
        requested_duration = duration_for_speed(reference_width, px_per_mm, speed)
        session_manager.set_duration_ms(requested_duration)

        actual_duration = int(session_manager.gui_setup.duration_ms)
        warnings: List[str] = []
        if actual_duration != requested_duration:
            warnings.append(
                "The requested duration was limited by the current sweep "
                f"({requested_duration} ms requested, {actual_duration} ms applied)."
            )
        if missing_geometry:
            warnings.append(
                "Sensitivity was not changed for groups without visible cells: "
                + ", ".join(missing_geometry)
                + "."
            )
        if ordered_widths[-1] - ordered_widths[0] > 1.0:
            warnings.append(
                "Selected groups have different cell widths. Duration is global, "
                "so sweep speed was calibrated to the median cell width."
            )
        if warnings:
            QMessageBox.warning(parent, "EEG grid", "\n\n".join(warnings))

    def run(self, session_manager, add_on_data_dir):
        gui_setup = session_manager.gui_setup
        if gui_setup is None:
            return False
        groups = [
            group
            for group in (gui_setup.channels_groups or [])
            if getattr(group, "is_shown", True)
            and not getattr(group, "is_auxiliary", False)
            and getattr(group, "channel_indexes", [])
        ]
        if not groups:
            QMessageBox.warning(None, "EEG grid", "No visible non-auxiliary channel groups.")
            return False

        try:
            if self._window is not None:
                self._window.close()
        except Exception as e:
            ephyr_logger().debug(str(e))
        self._window = _EEGGridWindow(self, session_manager, Path(add_on_data_dir), groups)
        self._window.show()
        self._window.raise_()
        self._window.activateWindow()
        return False

    def view(
        self,
        add_on_data_dir: Path,
        processed_data,
        voltage_scale,
        start_point,
        duration_ms,
        start_time_ms,
        end_time_ms,
        sample_rate,
        axis_duration_ms,
        sweep_idx,
        visible_channel_indexes,
        channel_names,
        visible_events,
        visible_periods,
        channel_groups,
        channels_setup,
        painter: QPainter,
        signal_widget: QWidget,
        channel_rects: List[Tuple[int, QRect]],
        signal_width,
        draw_area_height,
        bg_color,
        grid_color,
        signal_color,
        text_color,
        axis_color,
    ):
        del (
            processed_data,
            voltage_scale,
            start_point,
            start_time_ms,
            end_time_ms,
            sample_rate,
            axis_duration_ms,
            sweep_idx,
            visible_channel_indexes,
            channel_names,
            visible_events,
            visible_periods,
            channels_setup,
            signal_widget,
            signal_width,
            draw_area_height,
            bg_color,
            grid_color,
            signal_color,
            text_color,
            axis_color,
        )
        if add_on_data_dir is None or duration_ms <= 0.0:
            return
        try:
            params = self.load_params(Path(add_on_data_dir))
            keys = params.get("target_group_keys", [])
            selected_groups = self.resolve_groups_by_keys(
                channel_groups,
                keys,
                non_aux_only=True,
                fallback_all=False,
            )
            selected_channels = {
                int(channel_idx)
                for group in selected_groups
                for channel_idx in (getattr(group, "channel_indexes", []) or [])
            }
            rects = [
                rect
                for channel_idx, rect in channel_rects
                if int(channel_idx) in selected_channels
            ]
            if not rects:
                return

            minor_pen = QPen(QColor(80, 80, 80), 1, Qt.PenStyle.DashLine)
            major_pen = QPen(QColor(80, 80, 80), 2, Qt.PenStyle.DashLine)
            for rect in rects:
                minor_x, major_x = grid_x_positions(rect, float(duration_ms))
                painter.setPen(minor_pen)
                for x in minor_x:
                    painter.drawLine(x, rect.top(), x, rect.bottom())
                painter.setPen(major_pen)
                for x in major_x:
                    painter.drawLine(x, rect.top(), x, rect.bottom())
        except Exception as exc:
            ephyr_logger().debug(str(exc))


__all__ = [
    "EEGGridAddOn",
    "duration_for_speed",
    "grid_x_positions",
    "pixels_per_mm",
    "voltage_scale_for_sensitivity",
]
