"""Spike importer add-on.

Runnable. Imports spikes detected outside Ephyr from a WEEGIT ``.spk`` file into
a spike set, so the viewer / navigation / aligned / raster add-ons can use them
exactly like spikes detected by ``spike_detection``.

Binary layout of a ``.spk`` file (as written by the original WEEGIT tooling)::

    offset   0 : method name, ASCII, NUL-padded to 50 bytes
    offset  50 : spike count N as ASCII decimal, NUL-padded to 50 bytes
    offset 100 : N x timestamp  (sample index inside the sweep, 1-based)
                 N x float32    amplitude (source units, magnitude of the peak)
                 N x int16      channel (1-based)
                 N x int16      sweep (1-based)
                 N x int16      cluster (0 = unclustered)

The timestamp array is ``float64`` in every known writer, while the format
description mentions ``float32``, so the dtype is taken from the file size.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PyQt6.QtWidgets import (
    QDialog,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
)

from ephyr.core.add_ons.base import BaseAddOn
from ephyr.logger import ephyr_logger

from ephyr_add_ons.spike_utils._common import (
    SOURCE_IMPORTED,
    SpikePoint,
    SpikeSetMeta,
    SpikesPayload,
    SpikeUtilsBase,
    safe_imported_set_dir_name,
)

SPK_FILTERS = "WEEGIT spike files (*.spk);;All files (*)"

_HEADER_BYTES = 100
_METHOD_FIELD_BYTES = 50
_TAIL_BYTES_PER_SPIKE = 4 + 2 + 2 + 2  # amplitude + channel + sweep + cluster


class SpkFormatError(Exception):
    """The file does not follow the WEEGIT .spk layout."""


@dataclass
class SpkContent:
    method_name: str
    timestamp_dtype: str
    timestamps: np.ndarray
    amplitudes: np.ndarray
    channels: np.ndarray
    sweeps: np.ndarray
    clusters: np.ndarray

    @property
    def count(self) -> int:
        return int(self.timestamps.size)


def _decode_ascii_field(raw: bytes) -> str:
    return raw.split(b"\x00")[0].decode("ascii", errors="replace").strip()


def read_spk_file(path: Path) -> SpkContent:
    path = Path(path)
    file_size = path.stat().st_size
    if file_size <= _HEADER_BYTES:
        raise SpkFormatError(f"The file is too small to be a .spk file ({file_size} bytes).")

    with open(path, "rb") as spk_file:
        header = spk_file.read(_HEADER_BYTES)

    method_name = _decode_ascii_field(header[:_METHOD_FIELD_BYTES]) or "unknown"
    count_text = _decode_ascii_field(header[_METHOD_FIELD_BYTES:_HEADER_BYTES])
    try:
        declared_count = int(float(count_text))
    except ValueError:
        raise SpkFormatError(
            f"The spike count field is not a number: {count_text!r}."
        )
    if declared_count <= 0:
        raise SpkFormatError(f"The file declares {declared_count} spikes, nothing to import.")

    payload_bytes = file_size - _HEADER_BYTES
    for dtype_name, timestamp_bytes in (("float64", 8), ("float32", 4)):
        if payload_bytes == declared_count * (timestamp_bytes + _TAIL_BYTES_PER_SPIKE):
            timestamp_dtype = dtype_name
            break
    else:
        raise SpkFormatError(
            f"The file declares {declared_count} spikes, but its data block is {payload_bytes} bytes, "
            f"which matches neither {declared_count * (8 + _TAIL_BYTES_PER_SPIKE)} bytes "
            f"(float64 timestamps) nor {declared_count * (4 + _TAIL_BYTES_PER_SPIKE)} bytes "
            f"(float32 timestamps)."
        )

    with open(path, "rb") as spk_file:
        spk_file.seek(_HEADER_BYTES)
        timestamps = np.fromfile(spk_file, dtype=timestamp_dtype, count=declared_count)
        amplitudes = np.fromfile(spk_file, dtype="float32", count=declared_count)
        channels = np.fromfile(spk_file, dtype="int16", count=declared_count)
        sweeps = np.fromfile(spk_file, dtype="int16", count=declared_count)
        clusters = np.fromfile(spk_file, dtype="int16", count=declared_count)

    for name, array in (
        ("timestamps", timestamps),
        ("amplitudes", amplitudes),
        ("channels", channels),
        ("sweeps", sweeps),
        ("clusters", clusters),
    ):
        if array.size != declared_count:
            raise SpkFormatError(
                f"The {name} array is truncated: {array.size} of {declared_count} values."
            )

    return SpkContent(
        method_name=method_name,
        timestamp_dtype=timestamp_dtype,
        timestamps=timestamps.astype(np.float64),
        amplitudes=amplitudes.astype(np.float64),
        channels=channels.astype(np.int64),
        sweeps=sweeps.astype(np.int64),
        clusters=clusters.astype(np.int64),
    )


def validate_against_header(content: SpkContent, header) -> List[str]:
    """Everything that makes the file unusable for this experiment."""
    problems: List[str] = []

    number_of_channels = int(header.number_of_channels)
    number_of_sweeps = int(header.number_of_sweeps)
    points_per_sweep = [int(points) for points in header.number_of_points_per_sweep]

    min_channel = int(content.channels.min())
    max_channel = int(content.channels.max())
    if min_channel < 1:
        problems.append(f"Channel numbers must be 1-based, but the file contains {min_channel}.")
    if max_channel > number_of_channels:
        problems.append(
            f"The file references channel {max_channel}, while the experiment has "
            f"{number_of_channels} channels."
        )

    min_sweep = int(content.sweeps.min())
    max_sweep = int(content.sweeps.max())
    if min_sweep < 1:
        problems.append(f"Sweep numbers must be 1-based, but the file contains {min_sweep}.")
    if max_sweep > number_of_sweeps:
        problems.append(
            f"The file references sweep {max_sweep}, while the experiment has "
            f"{number_of_sweeps} sweep(s)."
        )

    sample_indexes = np.rint(content.timestamps).astype(np.int64) - 1
    if int(sample_indexes.min()) < 0:
        problems.append(
            "Timestamps must be 1-based sample indexes, but the file contains "
            f"{float(content.timestamps.min())}."
        )
    for sweep_number in np.unique(content.sweeps):
        sweep_idx = int(sweep_number) - 1
        if not (0 <= sweep_idx < len(points_per_sweep)):
            continue
        sweep_mask = content.sweeps == sweep_number
        largest = int(sample_indexes[sweep_mask].max())
        sweep_points = points_per_sweep[sweep_idx]
        if largest >= sweep_points:
            problems.append(
                f"Sweep {int(sweep_number)}: a spike sits at sample {largest}, "
                f"while the sweep has only {sweep_points} points."
            )

    return problems


class SpikeImporterAddOn(SpikeUtilsBase, BaseAddOn):
    TRANSFORMATION = False
    VIEWABLE = False
    RUNNABLE = True

    def _ask_parameters(self, add_on_data_dir: Path) -> Optional[dict]:
        params = self.load_params(add_on_data_dir)
        dialog = QDialog()
        dialog.setWindowTitle("Spike importer")
        dialog.setMinimumWidth(560)
        outer = QVBoxLayout(dialog)
        form = QFormLayout()

        path_row = QHBoxLayout()
        path_edit = QLineEdit(str(params.get("spk_path", "")))
        path_edit.setPlaceholderText("Select a WEEGIT .spk file…")
        browse_btn = QPushButton("Browse…")

        def _browse() -> None:
            start = path_edit.text().strip() or str(Path.home())
            path, _ = QFileDialog.getOpenFileName(dialog, "Select .spk file", start, SPK_FILTERS)
            if path:
                path_edit.setText(path)

        browse_btn.clicked.connect(_browse)
        path_row.addWidget(path_edit, stretch=1)
        path_row.addWidget(browse_btn)
        form.addRow("Spike file:", path_row)
        form.addRow(
            QLabel(
                "Spikes are imported into a new spike set with the method name and the clusters\n"
                "stored in the file. Channels, sweeps and sample indexes are 1-based in .spk."
            )
        )
        outer.addLayout(form)

        actions = QHBoxLayout()
        btn_cancel = QPushButton("Cancel")
        btn_run = QPushButton("Import")
        actions.addStretch(1)
        actions.addWidget(btn_cancel)
        actions.addWidget(btn_run)
        outer.addLayout(actions)
        btn_cancel.clicked.connect(dialog.reject)
        btn_run.clicked.connect(dialog.accept)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return None

        spk_path = Path(path_edit.text().strip())
        if not path_edit.text().strip() or not spk_path.is_file():
            QMessageBox.warning(None, "Spike importer", "Select an existing .spk file.")
            return None

        self.save_params(add_on_data_dir, {"spk_path": str(spk_path)})
        return {"spk_path": spk_path}

    @staticmethod
    def _payloads_by_sweep(
        content: SpkContent,
        header,
        spk_path: Path,
    ) -> Tuple[Dict[int, SpikesPayload], int]:
        sample_rate = float(header.sample_rate)
        sample_indexes = np.rint(content.timestamps).astype(np.int64) - 1
        payloads: Dict[int, SpikesPayload] = {}
        imported = 0

        for sweep_number in np.unique(content.sweeps):
            sweep_idx = int(sweep_number) - 1
            sweep_mask = content.sweeps == sweep_number
            spikes_by_channel: Dict[int, List[SpikePoint]] = {}
            for channel_number in np.unique(content.channels[sweep_mask]):
                channel_mask = sweep_mask & (content.channels == channel_number)
                order = np.argsort(sample_indexes[channel_mask], kind="stable")
                channel_samples = sample_indexes[channel_mask][order]
                channel_amps = content.amplitudes[channel_mask][order]
                channel_clusters = content.clusters[channel_mask][order]
                spikes = [
                    SpikePoint(
                        sample_idx=int(sample_idx),
                        time_ms=float(sample_idx) / sample_rate * 1000.0,
                        # WEEGIT stores the magnitude of a negative peak.
                        value=-abs(float(amplitude)),
                        polarity="negative",
                        cluster=int(cluster),
                    )
                    for sample_idx, amplitude, cluster in zip(
                        channel_samples, channel_amps, channel_clusters
                    )
                ]
                spikes_by_channel[int(channel_number) - 1] = spikes
                imported += len(spikes)

            payloads[sweep_idx] = SpikesPayload(
                detector_name=content.method_name,
                source=SOURCE_IMPORTED,
                threshold=0.0,
                sweep_idx=sweep_idx,
                sample_rate=sample_rate,
                adaptive_sigma=False,
                detect_positive=False,
                detect_negative=True,
                merge_window_ms=0.0,
                source_file=str(spk_path),
                spikes_by_channel=spikes_by_channel,
            )
        return payloads, imported

    def run(self, session_manager, add_on_data_dir):
        header = session_manager.header
        add_on_data_dir = Path(add_on_data_dir)
        params = self._ask_parameters(add_on_data_dir)
        if params is None:
            return

        spk_path: Path = params["spk_path"]
        yield {"progress": 5, "message": f"Reading {spk_path.name}…"}
        try:
            content = read_spk_file(spk_path)
        except Exception as exc:
            ephyr_logger().error(f"Failed to read {spk_path}", exc_info=exc)
            QMessageBox.critical(None, "Spike importer", f"Could not read {spk_path.name}:\n\n{exc}")
            return

        problems = validate_against_header(content, header)
        if problems:
            QMessageBox.critical(
                None,
                "Spike importer",
                f"{spk_path.name} does not match this experiment, nothing was imported:\n\n"
                + "\n".join(f"• {problem}" for problem in problems),
            )
            return

        # A different recording name is suspicious but legitimate (renamed files).
        recording_name = str(getattr(header, "name_before_conversion", "") or "").strip()
        spk_stem = spk_path.name[: -len(".spk")] if spk_path.name.endswith(".spk") else spk_path.stem
        if recording_name and spk_stem and Path(recording_name).stem != spk_stem:
            answer = QMessageBox.question(
                None,
                "Spike importer",
                f"The file is named '{spk_stem}', while the experiment was converted from "
                f"'{recording_name}'.\n\nImport anyway?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if answer != QMessageBox.StandardButton.Yes:
                return

        yield {"progress": 40, "message": f"Mapping {content.count} spikes…"}
        payloads, imported = self._payloads_by_sweep(content, header, spk_path)
        if not payloads:
            QMessageBox.warning(None, "Spike importer", "The file contains no spikes to import.")
            return

        out_dir = self.spike_sets_dir(add_on_data_dir) / safe_imported_set_dir_name(
            spk_path.name, content.method_name
        )
        clusters = sorted({int(cluster) for cluster in np.unique(content.clusters)})
        try:
            for done, (sweep_idx, payload) in enumerate(sorted(payloads.items()), start=1):
                self.save_spikes_payload(out_dir / f"{sweep_idx}.spikes.json", payload)
                yield {
                    "progress": int(40 + (done / len(payloads)) * 55),
                    "message": f"Saved sweep {sweep_idx} ({done}/{len(payloads)})",
                }
            self.save_spike_set_meta(
                out_dir,
                SpikeSetMeta(
                    source=SOURCE_IMPORTED,
                    detector_name=content.method_name,
                    clusters=clusters,
                    source_file=str(spk_path),
                    created_at=datetime.now().isoformat(timespec="seconds"),
                ),
            )
        except Exception as exc:
            ephyr_logger().error(f"Failed to save the spike set from {spk_path}", exc_info=exc)
            QMessageBox.critical(None, "Spike importer", f"Could not save the spike set:\n\n{exc}")
            return

        real_clusters = [cluster for cluster in clusters if cluster != 0]
        ephyr_logger().info(
            f"Imported {imported} spikes from {spk_path} "
            f"(method '{content.method_name}', {content.timestamp_dtype} timestamps, "
            f"{len(payloads)} sweep(s), {len(real_clusters)} cluster(s)) into {out_dir}"
        )
        yield {
            "progress": 100,
            "message": f"Imported {imported} spikes into {out_dir.name}",
        }
