from __future__ import division

import scipy as sp

from guiqwt.builder import make
from guiqwt.baseplot import BasePlot
from guiqwt.plot import BaseCurveWidget

from ..spyke_exception import SpykeException
from dialogs import PlotDialog
import helper

@helper.needs_qt
def plot_signal(signal, events=None, epochs=None, spike_trains=None):
    """ Create a plot from a signal
    """
    if signal is None:
        raise SpykeException('Cannot create signal plot: No signal data provided!')
    if events is None:
        events = []
    if epochs is None:
        epochs = []
    if spike_trains is None:
        spike_trains = []

    # X-Axis
    sample = (1 / signal.sampling_rate).simplified
    x = sp.arange(signal.shape[0]) * sample

    # Plot title
    win_title = 'Analog Signal'
    if signal.recordingchannel:
        win_title += ' | Recording Channel: %s' %\
                     signal.recordingchannel.name
    if signal.segment:
        win_title += ' | Segment: %s' %\
                     signal.segment.name
    win = PlotDialog(toolbar=True, wintitle=win_title)


    pW = BaseCurveWidget(win)
    plot = pW.plot
    helper.add_epochs(plot, epochs, x.units)

    plot.add_item(make.curve(x, signal))
    #for s in spike_trains:
    #    self._add_templates(plot, spikes[u], templates[u][:, c] + offset, spike_offsets[u], self._get_color(u))

    helper.add_events(plot, events, x.units)
    #for s in spike_trains:
    #    self._add_spikes(plot, spikes[u], self._get_color(u))

    win.add_plot_widget(pW, 0)

    plot.set_axis_title(BasePlot.X_BOTTOM, 'Time')
    plot.set_axis_unit(BasePlot.X_BOTTOM, x.dimensionality.string)
    #plot.set_axis_title(BasePlot.Y_LEFT, 'Voltage')
    plot.set_axis_unit(BasePlot.Y_LEFT, signal.dimensionality.string)

    win.add_custom_curve_tools(False)

    #self._make_unit_legend(win, units)
    win.show()


@helper.needs_qt
def plot_signal_array(signalarray, events=None, epochs=None, spike_trains=None, plot_separate=True):
    """ Create a plot dialog from a signal array """
    if signalarray is None:
        raise SpykeException('Cannot create signal plot: No signal data provided!')
    if events is None:
        events = []
    if epochs is None:
        epochs = []
    if spike_trains is None:
        spike_trains = []

    # X-Axis
    sample = (1 / signalarray.sampling_rate).simplified
    x = sp.arange(signalarray.shape[0]) * sample

    # Plot title
    win_title = 'Analog Signals'
    if signalarray.recordingchannelgroup:
        win_title += ' | Recording Channel Group: %s' % \
                     signalarray.recordingchannelgroup.name
    if signalarray.segment:
        win_title += ' | Segment: %s' %\
                     signalarray.segment.name
    win = PlotDialog(toolbar=True, wintitle=win_title)


    channels = range(signalarray.shape[1])
    if plot_separate:
        plot = None
        for c in channels:
            pW = BaseCurveWidget(win)
            plot = pW.plot

            helper.add_epochs(plot, epochs, x.units)
            plot.add_item(make.curve(x, signalarray[:, c]))
            helper.add_events(plot, events, x.units)
            #for s in spike_trains:
            #    if show_templates:
            #        self._add_templates(pl, spikes[u], templates[u][:, c], spike_offsets[u], self._get_color(u))
            #    if show_spikes:
            #        self._add_spikes(pl, spikes[u], self._get_color(u))
            win.add_plot_widget(pW, c)
            #plot.set_axis_title(BasePlot.Y_LEFT, 'Voltage')
            plot.set_axis_unit(BasePlot.Y_LEFT, signalarray.dimensionality.string)

        plot.set_axis_title(BasePlot.X_BOTTOM, 'Time')
        plot.set_axis_unit(BasePlot.X_BOTTOM, x.dimensionality.string)

        win.add_x_synchronization_option(True, channels)
        win.add_y_synchronization_option(False, channels)
    else:
        channels.reverse()

        pW = BaseCurveWidget(win)
        plot = pW.plot

        helper.add_epochs(plot, epochs, x.units)

        # Find plot y offset
        maxOffset = 0
        for i, c in enumerate(channels[1:], 1):
            offset = signalarray[:, channels[i - 1]].max() - \
                     signalarray[:, c].min()
            if offset > maxOffset:
                maxOffset = offset

        offset = 0

        offset -= signalarray[:, channels[0]].min()

        for c in channels:
            plot.add_item(make.curve(x, signalarray[:, c] + offset))
            #for s in spike_trains:
            #    self._add_templates(plot, spikes[u], templates[u][:, c] + offset, spike_offsets[u], self._get_color(u))
            offset += maxOffset
        helper.add_events(plot, events, x.units)
        #for s in spike_trains:
        #    self._add_spikes(plot, spikes[u], self._get_color(u))

        win.add_plot_widget(pW, 0)

        plot.set_axis_title(BasePlot.X_BOTTOM, 'Time')
        plot.set_axis_unit(BasePlot.X_BOTTOM, x.dimensionality.string)
        #plot.set_axis_title(BasePlot.Y_LEFT, 'Voltage')
        plot.set_axis_unit(BasePlot.Y_LEFT, signalarray.dimensionality.string)

    win.add_custom_curve_tools(False)

    #self._make_unit_legend(win, units)
    win.show()