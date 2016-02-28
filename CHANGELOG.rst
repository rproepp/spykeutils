Version 0.4.3
-------------

* Fixes to work with numpy 1.11 and scipy 0.17

Version 0.4.2
-------------

* Integration of pymuvr (when available) for faster calculation of van Rossum
  distance.

Version 0.4.1
-------------
* Faster caching for Neo lazy loading.
* Faster correlogram calculation.

Version 0.4.0
-------------
* Correlogram plot supports new square plot matrix mode and count per second
  in addition to per segment.
* New options in spike waveform plot.
* DataProvider objects support transparent lazy loading for compatible IOs
  (currently only Hdf5IO).
* DataProvider can be forced to use a certain IO instead of automatically
  determining it by file extension.
* Load parameters for IOs can be specified in DataProvider.
* IO class, IO parameters and IO plugins are saved in selections and properly
  used in startplugin.py
* Qt implementation of ProgressBar available in plot.helper (moved from
  Spyke Viewer).
* Loading support for IO plugins (moved from Spyke Viewer).

Version 0.3.0
-------------
* Added implementations for various spike train metrics.
* Added generation functions for poisson spike trains
* Added tools module with various utility functions, e.g. binning
  spike trains or removing objects from Neo hierarchies.
* Added explained variance function to spike sorting quality assessment.
* Improved legends for plots involving colored lines.
* Plots now have a minimum size and scroll bars appear if the plots would
  become too small.
* Renamed plot.ISI to plot.isi for consistency

Version 0.2.1
-------------
* Added "Home" and "Pan" tools for plots (useful when no middle mouse button
  is available).
* Changed default grid in plots to show only major grid.
* Added a method to DataProvider for refreshing views after object hierarchy
  changed.
* New parameter for DataProvider AnalogSignal methods: AnalogSignalArrays can
  be automatically converted and returned.
* Significantly improved speed of spike density estimation and optimal kernel
  size calculation.
* Spike sorting quality assessment using gaussian clusters is now possible
  without prewhitening spikes or providing prewhitened means.
* Renamed "spyke-plugin" script to "spykeplugin"

Version 0.2.0
-------------
Initial documented public release.
