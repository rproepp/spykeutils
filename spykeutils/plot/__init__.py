""" This package contains various plotting functions for neo objects.
The plots are created using guiqwt - if guiqwt is not installed, this
package can not be used.

:mod:`rasterplot` Module
-------------------------

.. automodule:: spykeutils.plot.rasterplot
    :members:
    :show-inheritance:

:mod:`analog_signals` Module
----------------------------

.. automodule:: spykeutils.plot.analog_signals
    :members:
    :show-inheritance:

:mod:`correlogram` Module
-------------------------

.. automodule:: spykeutils.plot.correlogram
    :members:
    :show-inheritance:

:mod:`interspike_intervals` Module
----------------------------------

.. automodule:: spykeutils.plot.interspike_intervals
    :members:
    :show-inheritance:

:mod:`sde` Module
-----------------

.. automodule:: spykeutils.plot.sde
    :members:
    :show-inheritance:

:mod:`psth` Module
------------------

.. automodule:: spykeutils.plot.psth
    :members:
    :show-inheritance:

:mod:`spike_waveforms` Module
------------------

.. automodule:: spykeutils.plot.spike_waveforms
    :members:
    :show-inheritance:

:mod:`spike_amp_hist` Module
------------------

.. automodule:: spykeutils.plot.spike_amplitude_histogram
    :members:
    :show-inheritance:

:mod:`dialogs` Module
---------------------

.. automodule:: spykeutils.plot.dialogs
    :members:
    :show-inheritance:

:mod:`helper` Module
--------------------

.. automodule:: spykeutils.plot.helper
    :members:
    :show-inheritance:
"""

from interspike_intervals import ISI
from dialog import PlotDialog
from rasterplot import raster_plot
from correlogram import cross_correlogram
from analog_signals import signal, signal_array
from psth import psth
from sde import sde
from spike_waveforms import spikes
from spike_amp_hist import spike_amplitude_histogram