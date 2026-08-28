# Spike utils

Toolkit for detecting, importing and inspecting spikes in Ephyr.

## Spike sets

Every tool in this package works with **spike sets**. A set is one directory in the experiment-wide `add_ons/data/spike_sets/` folder (`dev_spike_sets/` for development add-ons), so any spike add-on can read a set no matter which add-on produced it:

```
add_ons/data/spike_sets/{set_name}/spike_set_meta.json
add_ons/data/spike_sets/{set_name}/{sweep_idx}.spikes.json
```

`spike_set_meta.json` describes the set: `source` (`detected` or `imported`), `detector_name`, the channel group and detection parameters for detected sets, the list of `clusters`, and `source_file` / `created_at` for imported sets. Each `{sweep_idx}.spikes.json` holds `spikes_by_channel`, mapping a channel index to spikes with `sample_idx`, `time_ms`, `value`, `polarity` and `cluster`.

The **method** identifies a whole set (the same method re-run with another threshold is still that method), while the **cluster** is a per-spike sorting label inside the set (`0` means unclustered). Tools that can show clusters do so only when a set contains more than one.

## Spike detection (`spike_detection`)

**Runnable.** Detects spikes on the full current sweep for the selected channels using a preprocessing pipeline and MAD-based thresholding (global or adaptive rolling sigma). Results are saved per sweep as a spike set with `source: detected`.

Channels are processed **one at a time** (load → preprocess → detect → release) so HD-MEA runs do not keep a full `(n_channels × n_samples)` float64 matrix in RAM.

### Run fields

| Field | Role |
|-------|------|
| Channel group / Channels | Non-auxiliary group and channels to detect on |
| Preprocessing pipeline | Pipeline applied before thresholding |
| Threshold multiplier | Multiplier of the MAD/sigma estimate |
| Use adaptive rolling sigma | Toggle adaptive vs global sigma |
| Sigma window / step (ms) | Rolling window and step for adaptive sigma |
| Sigma floor (uV) | Lower bound for the sigma estimate |
| Sigma smooth windows | Smoothing length over rolling sigma windows |
| Detect downward / upward spikes | Polarity selection |
| Merge window (ms) | Merge nearby detections into one spike |
| Select events / Events action / Window before & after (ms) | Include or ignore samples around events |
| Select periods / Periods action | Include or ignore named periods |

## Spike importer (`spike_importer`)

**Runnable.** Imports spikes detected outside Ephyr from a WEEGIT `.spk` file into a spike set with `source: imported`, preserving the method name and the per-spike clusters. The set then behaves like a detected one in the viewer, navigation, aligned and raster tools.

### Run fields

| Field | Role |
|-------|------|
| Spike file | WEEGIT `.spk` file to import |

### `.spk` binary layout

| Offset | Content |
|--------|---------|
| 0 | Method name, ASCII, NUL-padded to 50 bytes (`abovestd`, `cluster`, `kilosort`, `SpyKING_CIRCUS`, …) |
| 50 | Spike count `N` as ASCII decimal, NUL-padded to 50 bytes |
| 100 | `N` timestamps (sample index inside the sweep, 1-based), `N` × `float32` amplitude, `N` × `int16` channel (1-based), `N` × `int16` sweep (1-based), `N` × `int16` cluster (`0` = unclustered) |

Timestamps are `float64` in every known writer while the original format note mentions `float32`, so the importer picks the dtype from the file size and rejects the file when neither matches.

### Mapping and validation

- `sample_idx = round(timestamp) - 1`, `time_ms = sample_idx / sample_rate × 1000`; channel and sweep numbers also become 0-based.
- Amplitudes are stored as `value = -abs(amplitude)` with `polarity: negative`, because WEEGIT records the magnitude of a negative peak.
- Nothing is written unless the file matches the experiment header. The import is aborted with the full list of problems when the data block size does not match the declared spike count, when a channel or sweep number is out of range, or when a spike sits beyond the end of its sweep.
- A file name that differs from the converted recording name is only a confirmable warning.

## Spike viewer (`spike_viewer`)

**Runnable + Viewable.** **Run** selects one or more spike sets. With **View** enabled, markers are overlaid on the signal panel: sets sharing a channel group get different colors and are stacked, and inside a set with several clusters each cluster gets its own color.

### Run fields

| Field | Role |
|-------|------|
| Spike sets | One or more saved spike sets |

## Spike navigation (`spike_navigation`)

**Runnable.** Opens a non-modal window to step through spikes for a chosen spike set, channel and (optionally) a single cluster. Selecting a spike recenters the signal view on it.

### Run fields (navigation window)

| Field | Role |
|-------|------|
| Spike set | Saved spike set |
| Channel | Channel within that set |
| Cluster | All clusters, or one cluster of a sorted set |
| Go to spike # | Jump to a spike index |
| Prev / Next / Go | Step or jump and recenter the view |

## Aligned spikes plot (`aligned_spikes_plot`)

**Runnable.** For selected channels and a time window, overlays spike waveforms aligned on spike time, one figure per channel. A set with several clusters gets one mean waveform per cluster, otherwise a single mean is drawn. Waveforms are taken from the signal reprocessed with the pipeline stored in the set.

### Run fields

| Field | Role |
|-------|------|
| Spike set | Saved spike set |
| Channels | Subset of channels from that set |
| Window from / to (ms) | Which spikes to include |
| Pre-spike / Post-spike (ms) | Waveform cutout around each spike |
| Middle line | Draw a vertical line at spike time |
| Background grid | Toggle axes grid |
| Plot image | Whether to open a matplotlib figure |

## Raster plot (`raster_plot`)

**Runnable.** Draws a horizontal raster (time on x, channels on y) for a chosen spike set and time window, independent of the on-screen channel layout. A set with several clusters is drawn with one color and legend entry per cluster.

### Run fields

| Field | Role |
|-------|------|
| Spike set | Saved spike set |
| Channels | Channels to show as raster rows |
| Window from / to (ms) | Time range of the raster |
| Plot image | Whether to open a matplotlib figure |
