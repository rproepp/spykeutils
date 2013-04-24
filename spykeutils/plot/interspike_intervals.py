import scipy as sp

from guiqwt.builder import make
from guiqwt.baseplot import BasePlot
from guiqwt.plot import BaseCurveWidget

import quantities as pq

from .. import SpykeException
from dialog import PlotDialog
import helper


@helper.needs_qt
def isi(trains, bin_size, cut_off, bar_plot=False, time_unit=pq.ms):
    """ Create a plot dialog with an interspike interval histogram.

    :param dict trains: Dictionary with lists of spike trains indexed by
        units for which to display ISI histograms
    :param bin_size: Bin size (time)
    :type bin_size: Quantity scalar
    :param cut_off: End of histogram (time)
    :type bin_size: Quantity scalar
    :param bool bar_plot: If ``True``, create a bar ISI histogram for each
        index in ``trains``. Else, create a line ISI histogram.
    :param Quantity time_unit: Unit of X-Axis.
    """
    if not trains:
        raise SpykeException('No spike trains for ISI histogram')

    win_title = 'ISI Histogram | Bin size: ' + str(bin_size)
    win = PlotDialog(toolbar=True, wintitle=win_title, min_plot_width=150,
                     min_plot_height=100)
    bin_size = bin_size.rescale(time_unit)
    cut_off = cut_off.rescale(time_unit)
    bins = sp.arange(0 * time_unit, cut_off, bin_size) * time_unit

    legends = []
    if bar_plot:
        ind = 0
        columns = int(sp.sqrt(len(trains)))
        for i, train_list in trains.iteritems():
            pW = BaseCurveWidget(win)
            plot = pW.plot
            intervals = []
            for t in train_list:
                t = t.rescale(time_unit)
                sTrain = sp.asarray(t)
                sTrain.sort()
                intervals.extend(sp.diff(sTrain))

            (isi, bins) = sp.histogram(intervals, bins)

            if i and hasattr(i, 'name') and i.name:
                name = i.name
            else:
                name = 'Unknown'

            show_isi = list(isi)
            show_isi.insert(0, show_isi[0])
            curve = make.curve(
                bins, show_isi, name, color='k',
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
                plot.set_axis_title(BasePlot.Y_LEFT, 'Number of intervals')
            if ind >= len(trains) - columns:
                plot.set_axis_title(BasePlot.X_BOTTOM, 'Interval length')
                plot.set_axis_unit(
                    BasePlot.X_BOTTOM, time_unit.dimensionality.string)

            win.add_plot_widget(pW, ind, column=ind % columns)
            ind += 1
    else:
        pW = BaseCurveWidget(win)
        plot = pW.plot
        legend_items = []

        for i, train_list in trains.iteritems():
            intervals = []
            for t in train_list:
                t = t.rescale(time_unit)
                sTrain = sp.asarray(t)
                sTrain.sort()
                intervals.extend(sp.diff(sTrain))

            (isi, bins) = sp.histogram(intervals, bins)

            if i and hasattr(i, 'name') and i.name:
                name = i.name
            else:
                name = 'Unknown'
            color = helper.get_object_color(i)

            curve = make.curve(bins, isi, name, color=color)
            legend_items.append(curve)
            plot.add_item(curve)

        win.add_plot_widget(pW, 0)

        legends.append(make.legend(restrict_items=legend_items))
        plot.add_item(legends[-1])

        plot.set_antialiasing(True)
        plot.set_axis_title(BasePlot.Y_LEFT, 'Number of intervals')
        plot.set_axis_title(BasePlot.X_BOTTOM, 'Interval length')
        plot.set_axis_unit(BasePlot.X_BOTTOM, time_unit.dimensionality.string)

    win.add_custom_curve_tools()
    win.add_legend_option(legends, True)
    win.show()

    if bar_plot and len(trains) > 1:
        win.add_x_synchronization_option(True, range(len(trains)))
        win.add_y_synchronization_option(False, range(len(trains)))

    return win