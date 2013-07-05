from guiqwt.builder import make
from guiqwt.baseplot import BasePlot
from guiqwt.plot import BaseCurveWidget
import quantities as pq
import scipy as sp

from .. import SpykeException
from ..progress_indicator import ProgressIndicator
from ..correlations import correlogram
from dialog import PlotDialog
import helper


@helper.needs_qt
def cross_correlogram(trains, bin_size, max_lag=500 * pq.ms,
                      border_correction=True, per_second=True,
                      square=False, time_unit=pq.ms, progress=None):
    """ Create (cross-)correlograms from a dictionary of spike train
    lists for different units.

    :param dict trains: Dictionary of :class:`neo.core.SpikeTrain` lists.
    :param bin_size: Bin size (time).
    :type bin_size: Quantity scalar
    :param max_lag: Maximum time lag for which spikes are considered
        (end time of calculated correlogram).
    :type max_lag: Quantity scalar
    :param bool border_correction: Apply correction for less data at higher
        timelags.
    :param bool per_second: If ``True``, the y-axis is count per second,
        otherwise it is count per spike train.
    :param bool square: If ``True``, the plot will include all
        cross-correlograms, even if they are just mirrored versions of each
        other. The autocorrelograms are displayed as the diagonal of a
        square plot matrix. If ``False``, mirrored plots are omitted.
    :param Quantity time_unit: Unit of X-Axis.
    :param progress: Set this parameter to report progress.
    :type progress: :class:`spykeutils.progress_indicator.ProgressIndicator`
    """
    if not trains:
        raise SpykeException('No spike trains for correlogram')
    if not progress:
        progress = ProgressIndicator()

    win_title = 'Correlogram | Bin size ' + str(bin_size)
    progress.begin('Creating correlogram')
    progress.set_status('Calculating...')
    win = PlotDialog(toolbar=True, wintitle=win_title, min_plot_width=150,
                     min_plot_height=100)

    correlograms, bins = correlogram(
        trains, bin_size, max_lag, border_correction,
        per_second, time_unit, progress)
    x = bins[:-1] + bin_size / 2

    crlgs = []
    indices = correlograms.keys()

    for i1 in xrange(len(indices)):
        start_i = 0
        if not square:
            start_i = i1
        for i2 in xrange(start_i, len(indices)):
            crlgs.append(
                (correlograms[indices[i1]][indices[i2]],
                 indices[i1], indices[i2]))

    columns = int(sp.sqrt(len(crlgs)))

    legends = []
    for i, c in enumerate(crlgs):
        legend_items = []
        pW = BaseCurveWidget(win)
        plot = pW.plot
        plot.set_antialiasing(True)
        plot.add_item(make.curve(x, c[0]))

        # Create legend
        color = helper.get_object_color(c[1])
        color_curve = make.curve(
            [], [], c[1].name,
            color, 'NoPen', linewidth=1, marker='Rect',
            markerfacecolor=color, markeredgecolor=color)
        legend_items.append(color_curve)
        plot.add_item(color_curve)
        if c[1] != c[2]:
            color = helper.get_object_color(c[2])
            color_curve = make.curve(
                [], [], c[2].name,
                color, 'NoPen', linewidth=1, marker='Rect',
                markerfacecolor=color, markeredgecolor=color)
            legend_items.append(color_curve)
            plot.add_item(color_curve)
        legends.append(make.legend(restrict_items=legend_items))
        plot.add_item(legends[-1])

        if i >= len(crlgs) - columns:
            plot.set_axis_title(BasePlot.X_BOTTOM, 'Time')
            plot.set_axis_unit(BasePlot.X_BOTTOM, time_unit.dimensionality.string)
        if i % columns == 0:
            plot.set_axis_title(BasePlot.Y_LEFT, 'Correlation')
            if per_second:
                plot.set_axis_unit(BasePlot.Y_LEFT, 'count/second')
            else:
                plot.set_axis_unit(BasePlot.Y_LEFT, 'count/segment')

        win.add_plot_widget(pW, i, column=i % columns)

    win.add_custom_curve_tools()
    progress.done()
    win.add_legend_option(legends, True)
    win.show()

    if len(crlgs) > 1:
        win.add_x_synchronization_option(True, range(len(crlgs)))
        win.add_y_synchronization_option(False, range(len(crlgs)))

    return win