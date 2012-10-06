"""
.. autofunction:: ISI(trains, bin_size, cut_off, bar_plot=False, unit=ms)
"""
import scipy as sp

from guiqwt.builder import make
from guiqwt.baseplot import BasePlot
from guiqwt.plot import BaseCurveWidget

import neo
import quantities as pq

from .. import SpykeException
from dialog import PlotDialog
import helper


@helper.needs_qt
def ISI(trains, bin_size, cut_off, bar_plot=False, unit=pq.ms):
    """ Create a plot dialog with a interspike interval histogram of units
    for a list of trials. Read required data from database.

    :param dict trains: Dictionary with lists of spike trains indexed by
        units for which to display ISI histograms
    :param bin_size: Bin size (time)
    :type bin_size: Quantity scalar
    :param cut_off: End of histogram (time)
    :type bin_size: Quantity scalar
    :param bool bar_plot: If ``True``, create a bar ISI histogram
        (automatically limits plot to just the first of the given units).
        Else, create a line ISI histogram.
    :param Quantity unit: Unit of X-Axis. If None, milliseconds are used.
    """
    if not trains:
        raise SpykeException('No spike trains for ISI histogram')

    win_title = 'ISI Histogram | Bin size: %.3f ms' % bin_size
    win = PlotDialog(toolbar=True, wintitle=win_title)
    bin_size.rescale(unit)
    cut_off.rescale(unit)
    bins = sp.arange(0*unit, cut_off, bin_size) * unit

    pW = BaseCurveWidget(win)
    plot = pW.plot
    legend_items = []
    if bar_plot:
        u = trains.keys()[0]
        if 'unique_id' in u.annotations:
            color = helper.get_color(u.annotations['unique_id'])
        else:
            color = helper.get_color(hash(u))
        win.add_unit_color(color)
    for u, train_list in trains.iteritems():
        intervals = []
        for t in train_list:
            t = t.rescale(unit)
            sTrain = sp.asarray(t)
            sTrain.sort()
            intervals.extend(sp.diff(sTrain))

        (isi, bins) = sp.histogram(intervals, bins)

        if isinstance(u, neo.Unit):
            color = helper.get_object_color(u)
            name = u.name
        else:
            name = 'No unit'
            color = 'k'

        if not bar_plot:
            curve = make.curve(bins, isi, name,
                color=color)
            legend_items.append(curve)
            plot.add_item(curve)
        else:
            show_isi = list(isi)
            show_isi.insert(0, show_isi[0])
            curve = make.curve(bins, show_isi, name, color='k',
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

    if bar_plot: # Rescale Bar ISI
        scale = plot.axisScaleDiv(BasePlot.Y_LEFT)
        plot.setAxisScale(BasePlot.Y_LEFT, 0, scale.upperBound())
        plot.set_antialiasing(False)
    else:
        plot.set_antialiasing(True)


if __name__ == '__main__':
    import guidata
    app = guidata.qapplication()

    unit1 = neo.Unit('Unit 1')
    unit1.annotate(unique_id=1)
    unit2 = neo.Unit('Unit 2')
    unit2.annotate(unique_id=2)

    samples = pq.UnitQuantity('samples', pq.s/32000.0, symbol='samples')
    #noinspection PyArgumentList
    train1 = neo.SpikeTrain(sp.arange(50)*10+sp.rand(50)*10, units='ms', t_stop=1000)
    train1.unit = unit1
    train2 = neo.SpikeTrain(sp.arange(20)*20*32, units=samples, t_stop=320000)
    train2.unit = unit2

    ISI({unit1:[train1], unit2:[train2]}, 1, 25, False)
    app.exec_()