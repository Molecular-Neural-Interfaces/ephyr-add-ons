# Neurophysiological utils

Clinical neurophysiology tools for [Ephyr](https://github.com/Molecular-Neural-Interfaces/ephyr).

## EEG grid

`eeg_grid` is a Runnable and Viewable add-on that:

- calibrates the visible time window to 15, 30, or 60 mm/s;
- calibrates selected channel groups to 3, 5, 7, 10, 15, or 20 µV/mm;
- draws a screen-fixed vertical grid with major one-second and minor 200 ms lines.

Calibration uses the physical DPI of the screen that currently contains the signal panel.
Run the add-on again after moving Ephyr to another monitor or materially resizing/rearranging
channel groups.
