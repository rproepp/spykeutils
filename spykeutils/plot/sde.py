"""
.. autofunction:: sde(trains, events=None, start=0 ms, stop=None, kernel_size=100 ms, optimize_steps=0, minimum_kernel=10 ms, maximum_kernel=500 ms, unit=ms, progress=None)
"""
import scipy as sp

from guiqwt.builder import make
from guiqwt.baseplot import BasePlot
from guiqwt.plot import BaseCurveWidget

import quantities as pq

from .. import SpykeException
from .. import rate_estimation
from ..progress_indicator import ProgressIndicator
from dialog import PlotDialog
import helper


@helper.needs_qt
def sde(trains, events=None, start=0*pq.ms, stop=None,
        kernel_size=100*pq.ms, optimize_steps=0,
        minimum_kernel=10*pq.ms, maximum_kernel=500*pq.ms,
        unit=pq.ms, progress=None):
    """ Create a spike density estimation plot.

    The spike density estimations give an estimate of the instantaneous
    rate. Optionally finds optimal kernel size for given data.

    :param dict trains: A dictionary of SpikeTrain lists.
    :param dict events: A dictionary (with the same indices as ``trains``)
        of Event objects or lists of Event objects. In case of lists,
        the first event in the list will be used for alignment. The events
        will be at time 0 on the plot. If None, spike trains will are used
        unmodified.
    :param start: The desired time for the start of the first bin. It
        will be recalculated if there are spike trains which start later
        than this time. This parameter can be negative (which could be
        useful when aligning on events).
    :type start: Quantity scalar
    :param stop: The desired time for the end of the last bin. It will
        be recalculated if there are spike trains which end earlier
        than this time.
    :type stop: Quantity scalar
    :param kernel_size: A uniform kernel size for all spike trains.
        Only used if optimization of kernel sizes is not used (i.e.
        ``optimize_steps`` is 0).
    :type kernel_size: Quantity scalar
    :param int optimize_steps: The number of different kernel sizes tried
        between ``minimum_kernel`` and ``maximum_kernel``.
        If 0, ``kernel_size`` will be used.
    :param minimum_kernel: The minimum kernel size to try in optimization.
    :type minimum_kernel: Quantity scalar
    :param maximum_kernel: The maximum kernel size to try in optimization.
    :type maximum_kernel: Quantity scalar
    :param Quantity unit: Unit of X-Axis.
    :param progress: Set this parameter to report progress.
    :type progress: :class:`spykeutils.progress_indicator.ProgressIndicator`
    """
    if not progress:
        progress = ProgressIndicator()

    start.units = unit
    if stop:
        stop.units = unit
    kernel_size.units = unit
    minimum_kernel.units = unit
    maximum_kernel.units = unit

    # Align spike trains
    for u in trains:
        if events:
            trains[u] = rate_estimation.aligned_spike_trains(
                trains[u], events)

    # Calculate spike density estimation
    if optimize_steps:
        steps = sp.logspace(sp.log10(minimum_kernel),
            sp.log10(maximum_kernel),
            optimize_steps) * unit
        sde, kernel_size, eval_points = \
            rate_estimation.spike_density_estimation(trains, start, stop,
                optimize_steps=steps, progress=progress)
    else:
        sde, kernel_size, eval_points = \
        rate_estimation.spike_density_estimation(trains, start, stop,
            kernel_size=kernel_size, progress=progress)
    progress.done()

    if not sde:
        raise SpykeException('No spike trains for SDE!')

    # Plot
    win_title = 'Kernel Density Estimation'
    win = PlotDialog(toolbar=True, wintitle=win_title)

    pW = BaseCurveWidget(win)
    plot = pW.plot
    plot.set_antialiasing(True)
    for u in trains:
        if u and u.name:
            name = u.name
        else:
            name = 'Unknown'

        curve = make.curve(eval_points, sde[u],
            title='%s, Kernel width %.2f %s' % (name, kernel_size[u],
                unit.dimensionality.string),
            color=helper.get_object_color(u))
        plot.add_item(curve)

    plot.set_axis_title(BasePlot.X_BOTTOM, 'Time')
    plot.set_axis_unit(BasePlot.X_BOTTOM, eval_points.dimensionality.string)
    plot.set_axis_title(BasePlot.Y_LEFT, 'Rate')
    plot.set_axis_unit(BasePlot.Y_LEFT, 'Hz')
    l = make.legend()
    plot.add_item(l)

    win.add_plot_widget(pW, 0)
    win.add_custom_curve_tools()
    win.add_legend_option([l], True)
    win.show()