"""
.. autoclass:: spykeutils.SpykeException

:mod:`conversions` Module
-------------------------

.. automodule:: spykeutils.conversions
    :members:
    :undoc-members:
    :show-inheritance:

:mod:`correlations` Module
--------------------------

.. automodule:: spykeutils.correlations
    :members:

:mod:`progress_indicator` Module
--------------------------------

.. automodule:: spykeutils.progress_indicator
    :members:
    :undoc-members:
    :show-inheritance:

:mod:`rate_estimation` Module
-----------------------------

.. automodule:: spykeutils.rate_estimation
    :members:

:mod:`signal_processing` Module
-------------------------------

.. automodule:: spykeutils.signal_processing
    :members:
    :show-inheritance:
    :undoc-members:

:mod:`spike_train_generation` Module
------------------------------------

.. automodule:: spykeutils.spike_train_generation
    :members:
    :undoc-members:

:mod:`spike_train_metrics` Module
------------------------------------

.. automodule:: spykeutils.spike_train_metrics
    :members:
    :undoc-members:

:mod:`sorting_quality_assesment` Module
---------------------------------------

.. automodule:: spykeutils.sorting_quality_assesment
    :members:
    :undoc-members:
    :show-inheritance:

:mod:`stationarity` Module
--------------------------

.. automodule:: spykeutils.stationarity
    :members:

:mod:`tools` Module
------------------------

.. automodule:: spykeutils.tools
    :members:
"""

__version__ = '0.4.3'


class SpykeException(Exception):
    """ Exception thrown when a function in spykeutils encounters a
        problem that is not covered by standard exceptions.

        When using Spyke Viewer, these exceptions will be caught and
        shown in the GUI, while general exceptions will not be caught
        (and therefore be visible in the console) for easier
        debugging.
    """
    pass
