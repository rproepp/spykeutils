"""
.. autofunction:: signals(signals, events=None, epochs=None, spike_trains=None, spikes=None, use_subplots=True, time_unit=s, y_unit=None, progress=None)
"""
from __future__ import division

import scipy as sp
import quantities as pq

from guiqwt.builder import make
from guiqwt.baseplot import BasePlot
from guiqwt.plot import BaseCurveWidget

from ..progress_indicator import ProgressIndicator
from .. import conversions
from .. import SpykeException
from dialog import PlotDialog
import helper


@helper.needs_qt
def signals(signals, events=None, epochs=None, spike_trains=None,
           spikes=None, show_waveforms=True, use_subplots=True,
           time_unit=pq.s, y_unit=None, progress=None):
    """ Create a plot from a list of AnalogSignal objects.

    :param list signals: The list of signals to plot.
    :param sequence events: A list of Event objects to be included in the
        plot.
    :param sequence epochs: A list of Epoch objects to be included in the
        plot.
    :param list spike_trains: A list of SpikeTrain objects to be
        included in the plot. The ``unit`` property (if it exists) is used
        for color and legend entries.
    :param list spikes: A list Spike objects to be included in the plot.
        The ``unit`` property (if it exists) is used for color and legend
        entries.
    :param bool show_waveforms: Determines if spikes from Spike and
        SpikeTrain objects are shown as waveforms or vertical lines.
    :param bool use_subplots: Determines if a separate subplot for is created
        each signal.
    :param Quantity time_unit: The unit of the x axis.
    :param progress: Set this parameter to report progress.
    :type progress: :class:`spykeutils.progress_indicator.ProgressIndicator`
    """
    if not signals:
        raise SpykeException(
            'Cannot create signal plot: No signal data provided!')
    if not progress:
        progress = ProgressIndicator()

    # Plot title
    win_title = 'Analog Signal'
    if len(set((s.recordingchannel for s in signals))) == 1:
        if signals[0].recordingchannel and signals[0].recordingchannel.name:
            win_title += ' | Recording Channel: %s' %\
                         signals[0].recordingchannel.name
    if len(set((s.segment for s in signals))) == 1:
        if signals[0].segment and signals[0].segment.name:
            win_title += ' | Segment: %s' % signals[0].segment.name
    win = PlotDialog(toolbar=True, wintitle=win_title)

    if events is None:
        events = []
    if epochs is None:
        epochs = []
    if spike_trains is None:
        spike_trains = []
    if spikes is None:
        spikes = []

    if show_waveforms:
        for st in spike_trains:
            if st.waveforms is not None:
                spikes.extend(conversions.spike_train_to_spikes(st))
        spike_trains = []
    else:
        unit_spikes = {}
        for s in spikes:
            unit_spikes.setdefault(s.unit, []).append(s)
        for sps in unit_spikes.itervalues():
            spike_trains.append(conversions.spikes_to_spike_train(sps, False))
        spikes = []

    channels = range(len(signals))

    progress.set_ticks((len(spike_trains) + len(spikes) + 1) * len(channels))

    offset = 0 * signals[0].units
    if use_subplots:
        plot = None
        for c in channels:
            pW = BaseCurveWidget(win)
            plot = pW.plot

            if signals[c].name:
                win.set_plot_title(plot, signals[c].name)
            elif signals[c].recordingchannel:
                if signals[c].recordingchannel.name:
                    win.set_plot_title(plot, signals[c].recordingchannel.name)

            sample = (1 / signals[c].sampling_rate).simplified
            x = (sp.arange(signals[c].shape[0])) * sample + signals[c].t_start
            x.units = time_unit

            helper.add_epochs(plot, epochs, x.units)
            if y_unit is not None:
                plot.add_item(make.curve(x, signals[c].rescale(y_unit)))
            else:
                plot.add_item(make.curve(x, signals[c]))
            helper.add_events(plot, events, x.units)

            _add_spike_waveforms(plot, spikes, x.units, c, offset, progress)

            for train in spike_trains:
                color = helper.get_object_color(train.unit)
                helper.add_spikes(plot, train, color, units=x.units)
                progress.step()

            win.add_plot_widget(pW, c)
            plot.set_axis_unit(BasePlot.Y_LEFT,
                signals[c].dimensionality.string)
            progress.step()

        plot.set_axis_title(BasePlot.X_BOTTOM, 'Time')
        plot.set_axis_unit(BasePlot.X_BOTTOM, x.dimensionality.string)
    else:
        channels.reverse()

        pW = BaseCurveWidget(win)
        plot = pW.plot

        helper.add_epochs(plot, epochs, time_unit)

        # Find plot y offset
        max_offset = 0 * signals[0].units
        for i, c in enumerate(channels[1:], 1):
            cur_offset = signals[channels[i - 1]].max() - signals[c].min()
            if cur_offset > max_offset:
                max_offset = cur_offset

        offset -= signals[channels[0]].min()

        for c in channels:
            sample = (1 / signals[c].sampling_rate).simplified
            x = (sp.arange(signals[c].shape[0])) * sample + signals[c].t_start
            x.units = time_unit

            if y_unit is not None:
                plot.add_item(make.curve(x,
                    (signals[c] + offset).rescale(y_unit)))
            else:
                plot.add_item(make.curve(x, signals[c] + offset))
            _add_spike_waveforms(plot, spikes, x.units, c, offset, progress)
            offset += max_offset
            progress.step()

        helper.add_events(plot, events, x.units)

        for train in spike_trains:
            color = helper.get_object_color(train.unit)
            helper.add_spikes(plot, train, color, units=x.units)
            progress.step()

        win.add_plot_widget(pW, 0)

        plot.set_axis_title(BasePlot.X_BOTTOM, 'Time')
        plot.set_axis_unit(BasePlot.X_BOTTOM, x.dimensionality.string)
        plot.set_axis_unit(BasePlot.Y_LEFT, signals[0].dimensionality.string)

    win.add_custom_curve_tools(False)

    units = set([s.unit for s in spike_trains])
    units = units.union([s.unit for s in spikes])

    progress.done()

    helper.make_window_legend(win, units, False)
    win.show()

    if use_subplots:
        win.add_x_synchronization_option(True, channels)
        win.add_y_synchronization_option(False, channels)


def _add_spike_waveforms(plot, spikes, x_units, channel, offset, progress):
    for spike in spikes:
        if spike.waveform is None:
            continue

        color = helper.get_object_color(spike.unit)
        # TODO: Is this usage of Spike.left_sweep correct?
        if spike.left_sweep:
            lsweep = spike.left_sweep
        else:
            lsweep = 0.0 * pq.ms
        start = (spike.time-lsweep).rescale(x_units)
        stop = (spike.waveform.shape[0] / spike.sampling_rate +
                spike.time - lsweep).rescale(x_units)
        spike_x = sp.arange(start, stop,
            (1.0 / spike.sampling_rate).rescale(x_units)) * x_units

        plot.add_item(make.curve(spike_x,
            spike.waveform[:, channel] + offset,
            color=color, linewidth=2))
        progress.step()