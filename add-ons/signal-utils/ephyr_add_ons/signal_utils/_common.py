"""Group-shared helpers for the Signal utils add-ons.

Only imports from ``ephyr.core.add_ons`` (shared infrastructure); it never
imports from another add-on group, keeping groups independent.
"""

from __future__ import annotations

from typing import Tuple

from ephyr.core.add_ons import EphyrAddOnMixin


def safe_file_name(raw: str) -> str:
    clean = (raw or "").strip()
    if not clean:
        return "channel"
    for ch in '<>:"/\\|?*':
        clean = clean.replace(ch, "_")
    return clean.replace(" ", "_")


class SignalUtilsBase(EphyrAddOnMixin):
    """Base for signal-domain add-ons with shared window resolution."""

    def sweep_duration_ms(self, session_manager, header) -> float:
        sample_rate = float(header.sample_rate)
        sweep_idx = int(session_manager.gui_setup.current_sweep_idx)
        sweep_points = int(header.number_of_points_per_sweep[sweep_idx])
        if sample_rate <= 0:
            return 0.0
        return (sweep_points / sample_rate) * 1000.0

    def default_window_ms(self, session_manager, header) -> Tuple[float, float]:
        sample_rate = float(header.sample_rate)
        start_ms = float(session_manager.gui_setup.start_point) * 1000.0 / sample_rate if sample_rate > 0 else 0.0
        return start_ms, start_ms + float(session_manager.gui_setup.duration_ms)

    def resolve_window_samples(
        self, session_manager, header, window_from_ms: float, window_to_ms: float
    ) -> Tuple[int, int, float, float, int]:
        """Clamp a millisecond window to valid sample indices for the current sweep."""
        sample_rate = float(header.sample_rate)
        sweep_idx = int(session_manager.gui_setup.current_sweep_idx)
        sweep_points = int(header.number_of_points_per_sweep[sweep_idx])
        start_sample = int(round((window_from_ms / 1000.0) * sample_rate))
        end_sample = int(round((window_to_ms / 1000.0) * sample_rate))
        start_sample = max(0, min(start_sample, sweep_points - 1))
        end_sample = max(start_sample + 1, min(end_sample, sweep_points))
        n_samples = max(1, end_sample - start_sample)
        start_second = start_sample / sample_rate
        end_second = end_sample / sample_rate
        return start_sample, end_sample, start_second, end_second, n_samples


__all__ = ["SignalUtilsBase", "safe_file_name"]
