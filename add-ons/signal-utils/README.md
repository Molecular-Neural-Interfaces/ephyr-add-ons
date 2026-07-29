# Signal utils

Add-ons for inspecting and comparing continuous signals in Ephyr.

Shared controls across plot add-ons:

| Field | Role |
|-------|------|
| Channel group / Channels | Non-auxiliary group and channels to analyze |
| Window from / to (ms) | Time window within the current sweep |
| Select events / Events action / Window before & after (ms) | Include or ignore samples around named events |
| Select periods / Periods action | Include or ignore named periods |
| Plot image | Whether to open a matplotlib figure after the run |

## Preprocessing plot (`preprocessing_plot`)

**Runnable.** For each selected channel, builds one figure with a subplot per selected preprocessing pipeline so pipelines can be compared on the same window.

### Extra Run fields

| Field | Role |
|-------|------|
| Pipelines | One or more saved pipelines to compare (includes `raw`) |
| Manage pipelines… | Opens the pipeline builder dialog |

## Power spectral density plot (`power_spectral_density_plot`)

**Runnable.** Computes Welch PSD for the selected channels and overlays them on a single plot.

### Extra Run fields

| Field | Role |
|-------|------|
| Pipeline | Preprocessing applied before PSD |
| Freq min / max (Hz) | Frequency band shown on the plot |
| Welch nperseg | Segment length for `scipy.signal.welch` |
| Overlap (%) | Segment overlap |
| Y scale | `dB (10*log10)` or `linear` |

## Spectrogram plot (`spectrogram_plot`)

**Runnable.** Draws an STFT time–frequency spectrogram for each selected channel.

### Extra Run fields

| Field | Role |
|-------|------|
| Pipeline | Preprocessing applied before the STFT |
| Freq min / max (Hz) | Frequency band of the spectrogram |
| STFT nperseg | Window length for `scipy.signal.spectrogram` |
| STFT overlap (%) | Window overlap |
