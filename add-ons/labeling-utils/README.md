# Labeling utils

Add-ons that help build and extend the annotation vocabulary in Ephyr.

## Events detection (`events_detection`)

**Runnable.** Scans the selected channels across all sweeps and appends detected markers to the session events vocabulary. If more than 1000 events are found, the user must confirm before they are added.

Two detection modes:

- **TTL** — threshold crossings on a rising or falling edge
- **Above threshold** — peak detection above a height (supports negative height), with a minimum inter-peak distance

### Run fields

| Field | Role                                                          |
|-------|---------------------------------------------------------------|
| Channel group / Channels | Group and channels to scan                                    |
| Preprocessing pipeline | Optional pipeline applied before detection (`raw` by default) |
| Detection mode | `TTL` or `Above threshold`                                    |
| Event name | Unique name written into the events vocabulary                |
| TTL edge | Rising or falling (TTL mode only)                             |
| TTL threshold | Crossing level (TTL mode only)                                |
| Height (threshold) | Peak height for above-threshold mode                          |
| Min distance (ms) | Minimum spacing between detections                            |

## Video sync (`video_sync`)

**Runnable.** Opens a non-modal window whose frame follows the center of the visible signal panel window. Choose the **source** up front:

1. **Video file (TTL ↔ flashes)** — hardware sync markers: TTL on an AUX channel matched to LED brightness flashes in an external video (`opencv-python`)
2. **NWB imaging by timestamps** — show `ImageSeries` / `OnePhotonSeries` / `TwoPhotonSeries` frames from an NWB file, aligned via `rate` / `starting_time` / `timestamps` (needs `pynwb`; no flash detection)

Use (1) only when imaging really contains sync flashes. Prefer (2) for multimodal NWB (e.g. ephys + GCaMP) that already shares a session clock.

### Video file pipeline

1. Detect TTL edges on the selected channel
2. Detect brightness flashes in the video
3. Cluster TTL pulses into trains by inter-pulse interval
4. Match a TTL train to the flash sequence (auto), or let the user pick manually
5. Fit an affine time map from matched pairs; outside the marker span the map is extrapolated

### NWB imaging pipeline

1. List imaging series in the NWB file
2. Read ephys and imaging `starting_time` (prefer the ElectricalSeries location from the converted header)
3. Build an identity map: `imaging_ms = signal_ms + (ephys_start − imaging_start) · 1000`
4. Open the sync window (TTL/flash UI hidden); optional manual pairs still refine the map

### Run fields

| Field | Role |
|-------|------|
| Source | `Video file (TTL ↔ flashes)` or `NWB imaging by timestamps` |
| Video file | Path to mp4/avi/mkv/mov (video mode) |
| Channel group / Channels | Group (often AUX) and TTL channel (first selected; video mode) |
| TTL edge / threshold / min distance | Same semantics as Events detection TTL mode |
| Flash sensitivity / min distance | Brightness peak detection in video |
| Sample fps | Frame subsample rate for flash scan (lower = faster) |
| Mode | `Auto match` or `Manual` (skip auto pairing; video mode) |
| NWB file | Path to `.nwb` with optical imaging (NWB mode; auto-guessed next to `*_ephyr` when possible) |
| Imaging series | Chosen `ImageSeries` / photon series location |

### Sync window

- Scrubbing the signal panel updates the frame (window center time)
- Status shows whether the current time is inside the **trusted** marker interval or **extrapolated**
- Video mode: pick which TTL train maps to flashes; Prev/Next TTL and flash buttons
- Both modes: add/remove pairs to refine the map; unlock scrub to browse frames freely
