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
