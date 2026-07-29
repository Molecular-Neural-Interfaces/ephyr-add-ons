"""Group-shared helpers for the Labeling utils add-ons."""

from __future__ import annotations

from ephyr.core.add_ons import EphyrAddOnMixin


class LabelingUtilsBase(EphyrAddOnMixin):
    """Base for labeling add-ons (event detection, ...)."""


__all__ = ["LabelingUtilsBase"]
