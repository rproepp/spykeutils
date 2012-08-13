import scipy as sp

from guiqwt.builder import make
from guiqwt.baseplot import BasePlot
from guiqwt.plot import BaseCurveWidget

import quantities as pq

from .. import rate_estimation
from ..spyke_exception import SpykeException
from dialogs import PlotDialog
import helper

def sde(trains, events, start, stop, kernel_size, progress,
        optimize_steps=0, minimum_kernel=10*pq.ms, maximum_kernel=500*pq.ms,
        unit=pq.ms):
    """ Create a spike density estimation plot.
    """
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
        else:
            trains[u] = trains[u].values()

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