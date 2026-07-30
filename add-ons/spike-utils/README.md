# Spike utils

Toolkit for detecting and inspecting spikes in Ephyr. Detection results are stored under the `spike_detection` add-on data directory and reused by the other tools in this package.

## Spike detection (`spike_detection`)

**Runnable.** Detects spikes on the full current sweep for the selected channels using a preprocessing pipeline and MAD-based thresholding (global or adaptive rolling sigma). Results are saved per sweep so viewer, navigation, aligned, and raster add-ons can read them.

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

## Spike viewer (`spike_viewer`)

**Runnable + Viewable.** **Run** selects one or more detection result sets. With **View** enabled, markers are overlaid on the signal panel (different colors when several methods share a group).

### Run fields

| Field | Role |
|-------|------|
| Detection methods | One or more saved detection result directories |

## Spike navigation (`spike_navigation`)

**Runnable.** Opens a non-modal window to step through spikes for a chosen detection set and channel. Selecting a spike recenters the signal view on it.

### Run fields (navigation window)

| Field | Role |
|-------|------|
| Detection method | Saved detection result set |
| Channel | Channel within that set |
| Go to spike # | Jump to a spike index |
| Prev / Next / Go | Step or jump and recenter the view |

## Aligned spikes plot (`aligned_spikes_plot`)

**Runnable.** For selected channels and a time window, overlays spike waveforms aligned on spike time (plus their mean), one figure per channel. Waveforms are taken from the signal reprocessed with the pipeline used for detection.

### Run fields

| Field | Role |
|-------|------|
| Detection method | Saved detection result set |
| Channels | Subset of channels from that detection |
| Window from / to (ms) | Which spikes to include |
| Pre-spike / Post-spike (ms) | Waveform cutout around each spike |
| Middle line | Draw a vertical line at spike time |
| Background grid | Toggle axes grid |
| Plot image | Whether to open a matplotlib figure |

## Raster plot (`raster_plot`)

**Runnable.** Draws a horizontal raster (time on x, channels on y) for a chosen detection set and time window, independent of the on-screen channel layout.

### Run fields

| Field | Role |
|-------|------|
| Detection method | Saved detection result set |
| Channels | Channels to show as raster rows |
| Window from / to (ms) | Time range of the raster |
| Plot image | Whether to open a matplotlib figure |
