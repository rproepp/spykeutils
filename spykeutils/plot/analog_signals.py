from __future__ import division

import scipy as sp
import neo
import quantities as pq

from guiqwt.builder import make
from guiqwt.baseplot import BasePlot
from guiqwt.plot import BaseCurveWidget

from ..spyke_exception import SpykeException
from .. import conversions
from ..progress_indicator import ProgressIndicator
from dialog import PlotDialog
import helper


@helper.needs_qt
def signal(signal, events=None, epochs=None, spike_trains=None,
           spike_waveforms=None, spike_train_waveforms = True,
           time_unit=pq.s, progress=ProgressIndicator()):
    """ Create a plot from an AnalogSignal.

    :param AnalogSignal signal: The signal to plot.
    :param sequence events: A list of Event objects to be included in the
        plot.
    :param sequence epochs: A list of Epoch objects to be included in the
        plot.
    :param dict spike_trains: A dictionary of SpikeTrain objects to be
        included in the plot. The way the spike trains are used depends on
    :param sequence spike_waveforms: A dictionary of lists of Spike objects
        to be included in the plot. Waveforms of spikes are overlaid on
        the signal. Indices of the dictionary (typically Unit objects) are
        used for color and legend entries.
    :param bool spike_train_waveforms: Determines how the spike trains from
        ``spike_trains`` are used:

        * ``True``: Spikes from the SpikeTrain objects are plotted in the same
          way as the spikes from ``spike_waveforms``. Only works if waveform
          data is present in the SpikeTrain objects.
        * ``False``: Spikes are plotted as vertical lines.
          Indices of the dictionary (typically Unit objects) are used
          for color and legend entries.
    :param Quantity time_unit: The unit of the x axis.
    :param progress: Set this parameter to report progress.
    :type progress: :class:`spykeutils.progress_indicator.ProgressIndicator`
    """
    # Plot title
    win_title = 'Analog Signal'
    if signal.recordingchannel:
        win_title += ' | Recording Channel: %s' % \
                     signal.recordingchannel.name
    if signal.segment:
        win_title += ' | Segment: %s' % signal.segment.name
    win = PlotDialog(toolbar=True, wintitle=win_title)

    signalarray = neo.AnalogSignalArray(sp.atleast_2d(sp.asarray(signal)).T,
        units=signal.units, sampling_rate=signal.sampling_rate)

    _plot_signal_array_on_window(win, signalarray, events, epochs,
        spike_trains, spike_waveforms, spike_train_waveforms, False,
        time_unit, progress)


@helper.needs_qt
def signal_array(signalarray, events=None, epochs=None, spike_trains=None,
                 spike_waveforms=None, spike_train_waveforms = True,
                 plot_separate=True, time_unit=pq.s,
                 progress=ProgressIndicator()):
    """ Create a plot dialog from an AnalogSignalArray.

    :param AnalogSignalArray signalarray: The signal array to plot.
    :param sequence events: A list of Event objects to be included in the
        plot.
    :param sequence epochs: A list of Epoch objects to be included in the
        plot.
    :param dict spike_trains: A dictionary of SpikeTrain objects to be
        included in the plot. Spikes are plotted as vertical lines.
        Indices of the dictionary (typically Unit objects) are used
        for color and legend entries.
    :param sequence spike_waveforms: A dictionary of lists of Spike objects
        to be included in the plot. Waveforms of spikes are overlaid on
        the signal. Indices of the dictionary (typically Unit objects) are
        used for color and legend entries.
    :param bool spike_train_waveforms: Determines how the spike trains from
        ``spike_trains`` are used:

        * ``True``: Spikes from the SpikeTrain objects are plotted in the same
          way as the spikes from ``spike_waveforms``. Only works if waveform
          data is present in the SpikeTrain objects.
        * ``False``: Spikes are plotted as vertical lines.
          Indices of the dictionary (typically Unit objects) are used
          for color and legend entries.
    :param bool plot_separate: Determines if a separate plot for is created
        each channel in ``signalarray``.
    :param Quantity time_unit: The unit of the x axis.
    :param progress: Set this parameter to report progress.
    :type progress: :class:`spykeutils.progress_indicator.ProgressIndicator`
    """
    # Plot title
    win_title = 'Analog Signals'
    if signalarray.recordingchannelgroup:
        win_title += ' | Recording Channel Group: %s' % \
                     signalarray.recordingchannelgroup.name
    if signalarray.segment:
        win_title += ' | Segment: %s' % signalarray.segment.name
    win = PlotDialog(toolbar=True, wintitle=win_title)

    _plot_signal_array_on_window(win, signalarray, events, epochs,
        spike_trains, spike_waveforms, spike_train_waveforms, plot_separate,
        time_unit, progress)


def _add_spike_waveforms(plot, spikes, x_units, channel, offset, progress):
    for spike in spikes:
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


def _plot_signal_array_on_window(win, signalarray, events=None, epochs=None,
                                 spike_trains=None, spikes=None,
                                 st_waveforms=False, plot_separate=True,
                                 time_unit=pq.s, progress=ProgressIndicator()):
    if signalarray is None:
        raise SpykeException(
            'Cannot create signal plot: No signal data provided!')
    if events is None:
        events = []
    if epochs is None:
        epochs = []
    if spike_trains is None:
        spike_trains = {}
    if spikes is None:
        spikes = {}

    if st_waveforms:
        draw_spikes = []
        for st in spike_trains:
            draw_spikes.extend(conversions.spikes_from_spike_train(st))
    else:
        draw_spikes = spikes

    channels = range(signalarray.shape[1])

    progress.set_ticks((len(draw_spikes) + len(spikes) + 1) * len(channels))

    # X-Axis
    sample = (1 / signalarray.sampling_rate).simplified
    x = sp.arange(signalarray.shape[0]) * sample
    x.units = time_unit

    offset = 0 * signalarray.units
    if plot_separate:
        plot = None
        for c in channels:
            pW = BaseCurveWidget(win)
            plot = pW.plot

            helper.add_epochs(plot, epochs, x.units)
            plot.add_item(make.curve(x, signalarray[:, c]))
            helper.add_events(plot, events, x.units)

            _add_spike_waveforms(plot, spikes, x.units, c, offset, progress)
            _add_spike_waveforms(plot, draw_spikes, x.units, c, offset,
                progress)

            if not st_waveforms:
                for train in spike_trains:
                    color = helper.get_object_color(train.unit)
                    helper.add_spikes(plot, train, color, units=x.units)

            win.add_plot_widget(pW, c)
            plot.set_axis_unit(BasePlot.Y_LEFT,
                signalarray.dimensionality.string)
            progress.step()

        plot.set_axis_title(BasePlot.X_BOTTOM, 'Time')
        plot.set_axis_unit(BasePlot.X_BOTTOM, x.dimensionality.string)
    else:
        channels.reverse()

        pW = BaseCurveWidget(win)
        plot = pW.plot

        helper.add_epochs(plot, epochs, x.units)

        # Find plot y offset
        max_offset = 0 * signalarray.units
        for i, c in enumerate(channels[1:], 1):
            cur_offset = signalarray[:, channels[i - 1]].max() -\
                         signalarray[:, c].min()
            if cur_offset > max_offset:
                max_offset = cur_offset

        offset -= signalarray[:, channels[0]].min()

        for c in channels:
            plot.add_item(make.curve(x, signalarray[:, c] + offset))
            _add_spike_waveforms(plot, spikes, x.units, c, offset, progress)
            _add_spike_waveforms(plot, draw_spikes, x.units, c, offset,
                progress)
            offset += max_offset
            progress.step()

        helper.add_events(plot, events, x.units)

        if not st_waveforms:
            for train in spike_trains:
                color = helper.get_object_color(train.unit)
                helper.add_spikes(plot, train, color, units=x.units)

        win.add_plot_widget(pW, 0)

        plot.set_axis_title(BasePlot.X_BOTTOM, 'Time')
        plot.set_axis_unit(BasePlot.X_BOTTOM, x.dimensionality.string)
        plot.set_axis_unit(BasePlot.Y_LEFT, signalarray.dimensionality.string)

    win.add_custom_curve_tools(False)

    units = set([s.unit for s in spike_trains])
    units = units.union([s.unit for s in spikes])

    progress.done()

    helper.make_window_legend(win, units, False)
    win.show()

    if plot_separate:
        win.add_x_synchronization_option(True, channels)
        win.add_y_synchronization_option(False, channels)