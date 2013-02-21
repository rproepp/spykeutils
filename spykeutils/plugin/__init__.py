""" This package provides support for writing plugins for Spyke Viewer.
It belongs to `spykeutils` so that plugins can be executed in an evironment
where the `spykeviewer` package and its dependencies are not installed
(e.g. servers).

`spykeutils` installs a script named "spykeplugin" that can be used to start
plugins directly from the command line, supplying selection and plugin
parameter information. It is also the default script that Spyke Viewer uses
when starting plugins remotely. If you want to implement your own script for
starting plugins remotely, e.g. on a server, you should conform to the
interface of this script.

:mod:`analysis_plugin` Module
-----------------------------

.. automodule:: spykeutils.plugin.analysis_plugin
    :members:
    :show-inheritance:

:mod:`data_provider` Module
---------------------------

.. automodule:: spykeutils.plugin.data_provider
    :members:
    :show-inheritance:

:mod:`gui_data` Module
----------------------

.. automodule:: spykeutils.plugin.gui_data
"""