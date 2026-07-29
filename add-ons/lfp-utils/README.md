# Local field potential utils

Add-ons for LFP analysis and visualization in Ephyr.

## Current-source density (`csd`)

**Viewable + Runnable.** Draws an interpolated CSD color field behind EEG traces for selected channel groups. CSD is computed on the fly in `view()`; **Run** only configures and persists display settings.

Enable **View** in the Add-ons side panel to see the overlay. Use **Run** to pick groups and scale.

### Run fields

| Field | Role |
|-------|------|
| Channel groups | Non-auxiliary groups that receive the CSD background |
| Scale min / Scale max | Manual color scale bounds for the CSD map |
| X pixel step | Horizontal downsampling step when rasterizing the overlay |
| Y pixel step | Vertical downsampling step when rasterizing the overlay |
| Reset to auto | Clears manual scale and returns to automatic vmin/vmax |
