import scipy as sp
import quantities as pq

from guiqwt.builder import make
from guiqwt.baseplot import BasePlot
from guiqwt.plot import BaseCurveWidget

from dialog import PlotDialog
import helper
from .. import rate_estimation
from ..spyke_exception import SpykeException
from ..progress_indicator import ProgressIndicator

@helper.needs_qt
def psth(trains, events=None, start=0*pq.ms, stop=None, bin_size=100*pq.ms,
         bar_plot=False, unit=pq.ms, progress=ProgressIndicator()):
    """ Create a peri stimulus time histogram.

    The peri stimulus time histogram gives an estimate of the instantaneous
    rate.

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
    :param bin_size: The bin size for the histogram.
    :type bin_size: Quantity scalar
    :param bool bar_plot: Determines if a bar plot (``True``) or a line
        plot (``False``) will be created. In case of a bar plot, only
        the first index in ``trains`` will be shown in the plot.
    :param Quantity unit: Unit of X-Axis.
    :param progress: Set this parameter to report progress.
    :type progress: :class:`spykeutils.progress_indicator.ProgressIndicator`
    """
    if not trains:
        raise SpykeException('No spike trains for PSTH!')

    if bar_plot:
        k = trains.keys()[0]
        trains = {k:trains[k]}

    # Align spike trains
    for u in trains:
        if events:
            trains[u] = rate_estimation.aligned_spike_trains(
                trains[u], events)
        else:
            trains[u] = trains[u].values()

    rates, bins = rate_estimation.psth(trains, bin_size, start=start,
        stop=stop)
    progress.done()

    if not psth:
        raise SpykeException('No spike trains for PSTH!')

    win_title = 'PSTH | Bin size %.2f %s' % (bin_size,
                                             unit.dimensionality.string)
    win = PlotDialog(toolbar=True, wintitle=win_title)

    if not bar_plot:
        bins = 0.5 * sp.diff(bins) + bins[:-1]

    pW = BaseCurveWidget(win)
    plot = pW.plot
    legend_items = []
    for i, r in rates.iteritems():
        if i and i.name:
            name = i.name
        else:
            name = 'Unknown'

        if not bar_plot:
            curve = make.curve(bins, r, name,
                color=helper.get_object_color(i))
            legend_items.append(curve)
            plot.add_item(curve)
        else:
            show_rates = list(r)
            show_rates.insert(0, show_rates[0])
            curve = make.curve(bins, show_rates, name, color='k',
                curvestyle="Steps", shade=1.0)
            plot.add_item(curve)
            break

    win.add_plot_widget(pW, 0)

    if not bar_plot:
        legend = make.legend(restrict_items=legend_items)
        plot.add_item(legend)
        win.add_legend_option([legend], True)

    plot.set_axis_title(BasePlot.Y_LEFT, 'Number of intervals')
    plot.set_axis_title(BasePlot.X_BOTTOM, 'Interval length')
    plot.set_axis_unit(BasePlot.X_BOTTOM, unit.dimensionality.string)
    win.add_custom_curve_tools()
    win.show()

    if bar_plot: # Rescale Bar plot
        scale = plot.axisScaleDiv(BasePlot.Y_LEFT)
        plot.setAxisScale(BasePlot.Y_LEFT, 0, scale.upperBound())
        plot.set_antialiasing(False)
    else:
        plot.set_antialiasing(True)