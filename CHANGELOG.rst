* Added implementations for various spike train metrics.
* Added tools module with various utility functions, e.g. binning
  spike trains or removing objects from Neo hierarchies.
* Added explained variance function to spike sorting quality assessment.
* Improved legends for plots involving colored lines.
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