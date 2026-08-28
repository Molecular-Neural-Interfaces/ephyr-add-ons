"""Spike viewer add-on.

Runnable + Viewable. ``run()`` opens a dialog to pick one or more spike sets,
detected or imported (each already bound to a channel group). ``view()`` overlays
markers for all selected sets: same-group sets use different colors and are
stacked upward; sets for different groups are drawn on their own groups. Within a
set that carries sorting labels, each cluster gets its own color.
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

# Distinct marker colors for overlapping sets on the same group.
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

# Distinct marker colors for the clusters of a sorted set.
_CLUSTER_COLORS = (
    QColor(220, 0, 0),
    QColor(0, 110, 220),
    QColor(0, 160, 60),
    QColor(230, 130, 0),
    QColor(150, 0, 200),
    QColor(0, 165, 165),
    QColor(210, 0, 130),
    QColor(120, 90, 40),
    QColor(90, 90, 90),
)


class SpikeViewerAddOn(SpikeUtilsBase, BaseAddOn):
    TRANSFORMATION = False
    VIEWABLE = True
    RUNNABLE = True
    Z_INDEX = 250

    def __init__(self):
        # Cache payloads per spike set dir: path -> (mtime, payload)
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
        selected = self.choose_spike_sets_dialog(
            "Spike viewer",
            add_on_data_dir,
            selected_dirs=self._selected_dirs_from_params(params),
            label="Spike sets:",
        )
        if selected is None:
            return

        selected_dirs = [str(path) for path in selected]
        self.save_params(add_on_data_dir, {"selected_dirs": selected_dirs})
        self._payload_cache.clear()

        sweep_idx = int(session_manager.gui_setup.current_sweep_idx)
        total_spikes = 0
        for path in selected:
            payload = self.read_spikes_payload(path, sweep_idx)
            if payload is not None:
                total_spikes += sum(len(v) for v in payload.spikes_by_channel.values())
        yield {
            "progress": 100,
            "message": f"Viewing {len(selected_dirs)} spike set(s), {total_spikes} spikes",
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

        # Within one group, stack sets upward with distinct colors.
        by_group: Dict[str, List[int]] = defaultdict(list)
        for set_idx, (_set_dir, payload) in enumerate(loaded):
            group_key = str(payload.group_key or "").strip() or f"__set_{set_idx}"
            by_group[group_key].append(set_idx)

        offset_rank: Dict[int, int] = {}
        for _group_key, indexes in by_group.items():
            for rank, set_idx in enumerate(indexes):
                offset_rank[set_idx] = rank

        marker_size = 4
        half = marker_size // 2
        stack_step = marker_size + 2

        for set_idx, (_set_dir, payload) in enumerate(loaded):
            set_color = _MARKER_COLORS[set_idx % len(_MARKER_COLORS)]
            # A sorted set gets one color per cluster; an unsorted one keeps the set color.
            cluster_colors = self._cluster_color_map(payload)
            y_shift = -offset_rank.get(set_idx, 0) * stack_step

            allowed_channels = set(
                self.channels_for_spikes_payload(payload, channel_groups=channel_groups)
            )
            allowed_channels.update(int(ch) for ch in (payload.spikes_by_channel or {}).keys())
            if not allowed_channels:
                continue

            painter.setPen(QPen(set_color))
            painter.setBrush(set_color)
            current_color = set_color
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
                    color = cluster_colors.get(int(spike.cluster), set_color) if cluster_colors else set_color
                    if color is not current_color:
                        painter.setPen(QPen(color))
                        painter.setBrush(color)
                        current_color = color
                    rel = (spike.time_ms - start_time_ms) / axis_duration_ms
                    x = int(rect.left() + rel * rect.width())
                    painter.drawRect(x - half, y - half, marker_size, marker_size)

    @staticmethod
    def _cluster_color_map(payload: SpikesPayload) -> Dict[int, QColor]:
        clusters = [cluster for cluster in payload.clusters() if cluster != 0]
        if len(clusters) < 2:
            return {}
        return {
            cluster: _CLUSTER_COLORS[idx % len(_CLUSTER_COLORS)]
            for idx, cluster in enumerate(clusters)
        }
