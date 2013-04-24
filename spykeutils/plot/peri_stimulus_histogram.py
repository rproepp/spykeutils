import scipy as sp
import quantities as pq

from guiqwt.builder import make
from guiqwt.baseplot import BasePlot
from guiqwt.plot import BaseCurveWidget

from .. import rate_estimation
from ..progress_indicator import ProgressIndicator
from .. import SpykeException
from dialog import PlotDialog
import helper


@helper.needs_qt
def psth(trains, events=None, start=0 * pq.ms, stop=None,
         bin_size=100 * pq.ms, rate_correction=True, bar_plot=False,
         time_unit=pq.ms, progress=None):
    """ Create a peri stimulus time histogram.

    The peri stimulus time histogram gives an estimate of the instantaneous
    rate.

    :param dict trains: A dictionary of :class:`neo.core.SpikeTrain` lists.
    :param dict events: A dictionary of Event objects, indexed by segment.
        The events will be at time 0 on the plot. If None, spike trains
        are used unmodified.
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
    :param bool rate_correction: Determines if a rates (``True``) or
        counts (``False``) are shown.
    :param bool bar_plot: Determines if a bar plot (``True``) or a line
        plot (``False``) will be created. In case of a bar plot, one plot
        for each index in ``trains`` will be created.
    :param Quantity time_unit: Unit of X-Axis.
    :param progress: Set this parameter to report progress.
    :type progress: :class:`spykeutils.progress_indicator.ProgressIndicator`
    """
    if not trains:
        raise SpykeException('No spike trains for PSTH!')
    if not progress:
        progress = ProgressIndicator()

    # Align spike trains
    for u in trains:
        if events:
            trains[u] = rate_estimation.aligned_spike_trains(
                trains[u], events)

    rates, bins = rate_estimation.psth(
        trains, bin_size, start=start, stop=stop,
        rate_correction=rate_correction)
    bins = bins.rescale(time_unit)

    if not psth:
        raise SpykeException('No spike trains for PSTH!')

    win_title = 'PSTH | Bin size %.2f %s' % (bin_size,
                                             time_unit.dimensionality.string)
    win = PlotDialog(toolbar=True, wintitle=win_title, min_plot_width=150,
                     min_plot_height=100)

    legends = []
    if bar_plot:
        ind = 0
        columns = int(sp.sqrt(len(rates)))
        for i, r in rates.iteritems():
            if i and hasattr(i, 'name') and i.name:
                name = i.name
            else:
                name = 'Unknown'

            pW = BaseCurveWidget(win)
            plot = pW.plot

            show_rates = list(r)
            show_rates.insert(0, show_rates[0])
            curve = make.curve(
                bins, show_rates, name, color='k',
                curvestyle="Steps", shade=1.0)
            plot.add_item(curve)

            # Create legend
            color = helper.get_object_color(i)
            color_curve = make.curve(
                [], [], name, color, 'NoPen', linewidth=1, marker='Rect',
                markerfacecolor=color, markeredgecolor=color)
            plot.add_item(color_curve)
            legends.append(make.legend(restrict_items=[color_curve]))
            plot.add_item(legends[-1])

            # Prepare plot
            plot.set_antialiasing(False)
            scale = plot.axisScaleDiv(BasePlot.Y_LEFT)
            plot.setAxisScale(BasePlot.Y_LEFT, 0, scale.upperBound())
            if ind % columns == 0:
                if not rate_correction:
                    plot.set_axis_title(BasePlot.Y_LEFT, 'Spike Count')
                else:
                    plot.set_axis_title(BasePlot.Y_LEFT, 'Rate')
                    plot.set_axis_unit(BasePlot.Y_LEFT, 'Hz')
            if ind >= len(trains) - columns:
                plot.set_axis_title(BasePlot.X_BOTTOM, 'Time')
                plot.set_axis_unit(
                    BasePlot.X_BOTTOM, time_unit.dimensionality.string)

            win.add_plot_widget(pW, ind, column=ind % columns)
            ind += 1
    else:
        bins = 0.5 * sp.diff(bins) + bins[:-1]

        pW = BaseCurveWidget(win)
        plot = pW.plot
        legend_items = []

        for i, r in rates.iteritems():
            if i and hasattr(i, 'name') and i.name:
                name = i.name
            else:
                name = 'Unknown'

            curve = make.curve(
                bins, r, name,
                color=helper.get_object_color(i))
            legend_items.append(curve)
            plot.add_item(curve)

        win.add_plot_widget(pW, 0)

        legends.append(make.legend(restrict_items=legend_items))
        plot.add_item(legends[-1])

        if not rate_correction:
            plot.set_axis_title(BasePlot.Y_LEFT, 'Spike Count')
        else:
            plot.set_axis_title(BasePlot.Y_LEFT, 'Rate')
            plot.set_axis_unit(BasePlot.Y_LEFT, 'Hz')
        plot.set_axis_title(BasePlot.X_BOTTOM, 'Time')
        plot.set_axis_unit(BasePlot.X_BOTTOM, time_unit.dimensionality.string)
        plot.set_antialiasing(True)

    win.add_custom_curve_tools()
    win.add_legend_option(legends, True)
    progress.done()
    win.show()

    if bar_plot and len(rates) > 1:
        win.add_x_synchronization_option(True, range(len(rates)))
        win.add_y_synchronization_option(False, range(len(rates)))

    return win