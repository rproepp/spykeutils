""" This package contains various plotting functions for neo objects.
The plots are created using :mod:`guiqwt` - if it is not installed,
this package can not be used.

.. automodule:: spykeutils.plot.rasterplot
    :members:

.. automodule:: spykeutils.plot.correlogram
    :members:

.. automodule:: spykeutils.plot.interspike_intervals
    :members:

.. automodule:: spykeutils.plot.peri_stimulus_histogram
    :members:

.. automodule:: spykeutils.plot.sde
    :members:

.. automodule:: spykeutils.plot.analog_signals
    :members:

.. automodule:: spykeutils.plot.spike_amp_hist
    :members:

.. automodule:: spykeutils.plot.spike_waveforms
    :members:

:mod:`dialog` Module
--------------------

.. automodule:: spykeutils.plot.dialog
    :members:
    :show-inheritance:

:mod:`helper` Module
--------------------

.. automodule:: spykeutils.plot.helper
    :members:

:mod:`guiqwt_tools` Module
--------------------------

.. automodule:: spykeutils.plot.guiqwt_tools
    :members:
    :show-inheritance:
"""

from interspike_intervals import isi
from dialog import PlotDialog
from rasterplot import raster
from correlogram import cross_correlogram
from analog_signals import signals
from peri_stimulus_histogram import psth
from sde import sde
from spike_waveforms import spikes
from spike_amp_hist import spike_amplitude_histogram

