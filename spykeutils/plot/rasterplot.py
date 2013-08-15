from guiqwt.builder import make
from guiqwt.baseplot import BasePlot
from guiqwt.plot import BaseCurveWidget

import quantities as pq

from .. import SpykeException
from dialog import PlotDialog
import helper


@helper.needs_qt
def raster(trains, time_unit=pq.ms, show_lines=True, events=None, epochs=None):
    """ Create a new plotting window with a rasterplot of spiketrains.

        :param dict trains: Dictionary of spike trains indexed by a
            Neo object (Unit or Segment).
        :param Quantity time_unit: Unit of X-Axis.
        :param bool show_lines: Determines if a horizontal line will be shown
            for each spike train.
        :param sequence events: A sequence of neo `Event` objects that will
            be marked on the plot.

    """
    if not trains:
        raise SpykeException('No spike trains for rasterplot')

    if not time_unit:
        time_unit = pq.ms

    win_title = 'Spike Trains'
    win = PlotDialog(toolbar=True, wintitle=win_title, major_grid=False)

    pW = BaseCurveWidget(win)
    plot = pW.plot

    if events is None:
        events = []
    if epochs is None:
        epochs = []

    offset = len(trains)
    legend_items = []
    for u, t in trains.iteritems():
        color = helper.get_object_color(u)

        train = helper.add_spikes(
            plot, t, color, 2, 21, offset, u.name, time_unit)

        if u.name:
            legend_items.append(train)
        if show_lines:
            plot.add_item(make.curve(
                [t.t_start.rescale(time_unit), t.t_stop.rescale(time_unit)],
                [offset, offset], color='k'))
        offset -= 1

    helper.add_epochs(plot, epochs, time_unit)
    helper.add_events(plot, events, time_unit)

    plot.set_axis_title(BasePlot.X_BOTTOM, 'Time')
    plot.set_axis_unit(BasePlot.X_BOTTOM, time_unit.dimensionality.string)

    win.add_plot_widget(pW, 0)

    legend = make.legend(restrict_items=legend_items)
    plot.add_item(legend)
    win.add_legend_option([legend], True)

    if len(trains) > 1:
        plot.set_axis_limits(BasePlot.Y_LEFT, 0.5, len(trains) + 0.5)

    win.add_custom_curve_tools()
    win.show()

    return win