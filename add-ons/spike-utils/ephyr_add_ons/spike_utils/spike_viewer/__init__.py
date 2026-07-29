"""Spike viewer add-on.

Runnable + Viewable. ``run()`` opens a dialog to pick one or more detection
methods (each already bound to a channel group). ``view()`` overlays markers for
all selected results: same-group methods use different colors and are stacked
upward; methods for different groups are drawn on their own groups.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PyQt6.QtGui import QColor, QPainter, QPen
from PyQt6.QtWidgets import QWidget

from ephyr.core.add_ons.base import BaseAddOn
from ephyr.logger import ephyr_logger

from ephyr_add_ons.spike_utils._common import SpikesPayload, SpikeUtilsBase

# Distinct marker colors for overlapping methods on the same group.
_MARKER_COLORS = (
    QColor(255, 0, 0),
    QColor(0, 140, 255),
    QColor(0, 180, 60),
    QColor(255, 140, 0),
    QColor(160, 0, 200),
    QColor(0, 180, 180),
    QColor(220, 0, 120),
    QColor(80, 80, 80),
)


class SpikeViewerAddOn(SpikeUtilsBase, BaseAddOn):
    TRANSFORMATION = False
    VIEWABLE = True
    RUNNABLE = True
    Z_INDEX = 250

    def __init__(self):
        # Cache payloads per detection dir: path -> (mtime, payload)
        self._payload_cache: Dict[str, Tuple[float, SpikesPayload]] = {}

    @staticmethod
    def _selected_dirs_from_params(params: Dict[str, Any]) -> List[str]:
        raw_dirs = params.get("selected_dirs")
        if isinstance(raw_dirs, list) and raw_dirs:
            return [str(p) for p in raw_dirs if p]
        single = str(params.get("selected_dir", "") or "")
        return [single] if single else []

    def run(self, session_manager, add_on_data_dir):
        add_on_data_dir = Path(add_on_data_dir)
        params = self.load_params(add_on_data_dir)
        selected = self.choose_result_dirs_dialog(
            "Spike viewer",
            add_on_data_dir,
            selected_dirs=self._selected_dirs_from_params(params),
            label="Detection methods:",
        )
        if selected is None:
            return

        selected_dirs = [str(path) for path in selected]
        self.save_params(add_on_data_dir, {"selected_dirs": selected_dirs})
        self._payload_cache.clear()

        sweep_idx = int(session_manager.gui_setup.current_sweep_idx)
        total_spikes = 0
        labels: List[str] = []
        for path in selected:
            payload = self.read_spikes_payload(path, sweep_idx)
            if payload is not None:
                total_spikes += sum(len(v) for v in payload.spikes_by_channel.values())
            labels.append(self.detection_result_label(path))
        yield {
            "progress": 100,
            "message": f"Viewing {len(selected_dirs)} method(s), {total_spikes} spikes",
        }

    def _load_payloads(
        self,
        selected_dirs: List[str],
        sweep_idx: int,
    ) -> List[Tuple[Path, SpikesPayload]]:
        loaded: List[Tuple[Path, SpikesPayload]] = []
        for selected_dir in selected_dirs:
            path = Path(selected_dir) / f"{int(sweep_idx)}.spikes.json"
            if not path.exists():
                continue
            try:
                mtime = path.stat().st_mtime
            except Exception as e:
                ephyr_logger().debug(str(e))
                continue
            cache_key = str(path)
            cached = self._payload_cache.get(cache_key)
            if cached is not None and cached[0] == mtime:
                loaded.append((Path(selected_dir), cached[1]))
                continue
            payload = self.read_spikes_payload(Path(selected_dir), int(sweep_idx))
            if payload is None:
                continue
            self._payload_cache[cache_key] = (mtime, payload)
            loaded.append((Path(selected_dir), payload))
        return loaded

    def view(
        self,
        add_on_data_dir: Path,
        processed_data: Dict[int, np.ndarray],
        voltage_scale: float,
        start_point: int,
        duration_ms: float,
        start_time_ms: float,
        end_time_ms: float,
        sample_rate: float,
        axis_duration_ms: float,
        sweep_idx: int,
        visible_channel_indexes: List[int],
        channel_names: List[str],
        visible_events: List[Any],
        visible_periods: List[Any],
        channel_groups: List[Any],
        channels_setup: Dict[int, Any],
        painter: QPainter,
        signal_widget: QWidget,
        channel_rects: List[Tuple[int, "Any"]],
        signal_width: int,
        draw_area_height: int,
        bg_color: QColor,
        grid_color: QColor,
        signal_color: QColor,
        text_color: QColor,
        axis_color: QColor,
        **kwargs,
    ):
        if painter is None or signal_widget is None or duration_ms <= 0 or axis_duration_ms <= 0:
            return
        add_on_data_dir = Path(add_on_data_dir) if add_on_data_dir is not None else None
        if add_on_data_dir is None:
            return

        params = self.load_params(add_on_data_dir)
        selected_dirs = self._selected_dirs_from_params(params)
        if not selected_dirs:
            return

        loaded = self._load_payloads(selected_dirs, int(sweep_idx))
        if not loaded:
            return

        # Within one group, stack methods upward with distinct colors.
        by_group: Dict[str, List[int]] = defaultdict(list)
        for method_idx, (_result_dir, payload) in enumerate(loaded):
            group_key = str(payload.group_key or "").strip() or f"__method_{method_idx}"
            by_group[group_key].append(method_idx)

        offset_rank: Dict[int, int] = {}
        for _group_key, indexes in by_group.items():
            for rank, method_idx in enumerate(indexes):
                offset_rank[method_idx] = rank

        marker_size = 4
        half = marker_size // 2
        stack_step = marker_size + 2

        for method_idx, (_result_dir, payload) in enumerate(loaded):
            color = _MARKER_COLORS[method_idx % len(_MARKER_COLORS)]
            y_shift = -offset_rank.get(method_idx, 0) * stack_step

            allowed_channels = set(
                self.channels_for_detection_payload(payload, channel_groups=channel_groups)
            )
            allowed_channels.update(int(ch) for ch in (payload.spikes_by_channel or {}).keys())
            if not allowed_channels:
                continue

            painter.setPen(QPen(color))
            painter.setBrush(color)
            for channel_idx, rect in channel_rects:
                ch = int(channel_idx)
                if ch not in allowed_channels:
                    continue
                spikes = payload.spikes_by_channel.get(ch)
                if not spikes:
                    continue
                y = rect.center().y() + y_shift
                for spike in spikes:
                    if not (start_time_ms <= spike.time_ms <= end_time_ms):
                        continue
                    rel = (spike.time_ms - start_time_ms) / axis_duration_ms
                    x = int(rect.left() + rel * rect.width())
                    painter.drawRect(x - half, y - half, marker_size, marker_size)
