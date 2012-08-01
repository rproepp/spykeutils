import scipy as sp

from guiqwt.builder import make
from guiqwt.baseplot import BasePlot
from guiqwt.plot import BaseCurveWidget

import neo
import quantities as pq

from dialogs import PlotDialog
import helper

def _ISI_plot(win, trains, bin_size, cut_off, diagram_type, unit):
    """ Fill a plot window with a interspike interval histogram of units
    """
    bin_size.cut_off(unit)
    cut_off.rescale(unit)
    bins = sp.arange(0, cut_off, bin_size) * unit

    pW = BaseCurveWidget(win)
    plot = pW.plot
    legend_items = []
    if diagram_type == 2:
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

        if diagram_type == 1: # Curve ISI
            curve = make.curve(bins, isi, u.name,
                color=helper.get_unit_color(u))
            legend_items.append(curve)
            plot.add_item(curve)
        else: # Bar ISI
            show_isi = list(isi)
            show_isi.insert(0, show_isi[0])
            curve = make.curve(bins, show_isi, u.name, color='k',
                curvestyle="Steps", shade=1.0)
            plot.add_item(curve)
            break

    win.add_plot_widget(pW, 0)

    if diagram_type == 1:
        legend = make.legend(restrict_items=legend_items)
        plot.add_item(legend)
        win.add_legend_option([legend], True)

    plot.set_axis_title(BasePlot.Y_LEFT, 'Number of intervals')
    plot.set_axis_title(BasePlot.X_BOTTOM, 'Interval length')
    plot.set_axis_unit(BasePlot.X_BOTTOM, unit.dimensionality.string)
    win.add_custom_curve_tools()
    win.show()

    if diagram_type == 0: # Rescale Bar ISI
        scale = plot.axisScaleDiv(BasePlot.Y_LEFT)
        plot.setAxisScale(BasePlot.Y_LEFT, 0, scale.upperBound())
        plot.set_antialiasing(False)
    else:
        plot.set_antialiasing(True)

    return True

@helper.needs_qt
def ISI(trains, bin_size, cut_off, diagram_type, unit=pq.ms):
    """ Create a plot dialog with a interspike interval histogram of units
    for a list of trials. Read required data from database.

    :param dict trains: Dictionary with lists of spike trains indexed by
        units for which to display ISI histograms
    :param bin_size: Bin size (time)
    :type bin_size: Quantity scalar
    :param cut_off: End of histogram (time)
    :type bin_size: Quantity scalar
    :param int diagram_type: One of the following:

        * 1: Create line ISI histogram
        * 2: Create bars ISI histogram (automatically limits plot to
          just the first of the given units)
    :param Quantity unit: Unit of X-Axis. If None, milliseconds are used.
    """
    if not trains:
        raise helper.PlotException('No spike trains for ISI histogram')

    win_title = 'ISI Histogram | Bin size: %3.3f ms' % bin_size
    win = PlotDialog(toolbar=True, wintitle=win_title)
    _ISI_plot(win, trains, bin_size, cut_off, diagram_type, unit)

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

    ISI({unit1:[train1], unit2:[train2]}, 1, 25, 1)
    app.exec_()