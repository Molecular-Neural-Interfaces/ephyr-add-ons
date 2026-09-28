# Ephyr add-ons

Official catalog of installable add-ons for **[Ephyr](https://github.com/Molecular-Neural-Interfaces/ephyr)**.

Ephyr is a lightweight yet powerful environment for labelling electrophysiological data. It combines an adaptive interface with intelligent performance scaling to match your machine’s resources, ensuring stable real-time operation even under heavy loads.

Fully open-source and built in Python, the platform provides a flexible API for post-annotation data access. Its add-on architecture lets you extend functionality seamlessly without modifying the core codebase.

Install packages from **Add-ons → Manage** in the Ephyr GUI. This repository is the source of the catalog (`index.json`) and of each package under `add-ons/`.

## Implemented add-ons

- [Labeling utils](add-ons/labeling-utils) — detect events and append them to the session vocabulary
- [Local field potential utils](add-ons/lfp-utils) — current-source density (CSD) visualization on the signal panel
- [Neurophysiological utils](add-ons/neurophysiological-utils) — calibrated clinical EEG scaling and timing grid
- [Signal utils](add-ons/signal-utils) — preprocessing comparison, power spectral density, and spectrograms
- [Spike utils](add-ons/spike-utils) — spike detection, viewing, navigation, aligned waveforms, and rasters
