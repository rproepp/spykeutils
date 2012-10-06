""" This package contains various plotting functions for neo objects.
The plots are created using :mod:`guiqwt` - if it is not installed,
this package can not be used.

.. automodule:: spykeutils.plot.rasterplot

.. automodule:: spykeutils.plot.correlogram

.. automodule:: spykeutils.plot.interspike_intervals

.. automodule:: spykeutils.plot.peri_stimulus_histogram

.. automodule:: spykeutils.plot.sde

.. automodule:: spykeutils.plot.analog_signals

.. automodule:: spykeutils.plot.spike_amp_hist

.. automodule:: spykeutils.plot.spike_waveforms

:mod:`dialog` Module
--------------------

.. automodule:: spykeutils.plot.dialog
    :members:
    :show-inheritance:

:mod:`helper` Module
--------------------

.. automodule:: spykeutils.plot.helper
    :members:
"""

from interspike_intervals import ISI
from dialog import PlotDialog
from rasterplot import raster
from correlogram import cross_correlogram
from analog_signals import signals
from peri_stimulus_histogram import psth
from sde import sde
from spike_waveforms import spikes
from spike_amp_hist import spike_amplitude_histogram

