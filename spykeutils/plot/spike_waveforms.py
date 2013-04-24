import scipy as sp
import quantities as pq

from guiqwt.builder import make
from guiqwt.baseplot import BasePlot
from guiqwt.plot import BaseCurveWidget

from ..progress_indicator import ProgressIndicator
from .. import SpykeException
from dialog import PlotDialog
import helper


@helper.needs_qt
def spikes(spikes, axes_style, anti_alias=False, time_unit=pq.ms,
           progress=None):
    """ Create a plot dialog with spike waveforms.

    :param dict spikes: A dictionary of spike lists.
    :param int axes_style: Plotting mode. The following values are possible:

        - 1: Show each channel in a seperate plot, split vertically.
        - 2: Show each channel in a separate plot, split horizontally.
        - 3: Show each key of ``spikes`` in a separate plot,
          channels are split vertically.
        - 4: Show each key of ``spikes`` in a separate plot,
          channels are split horizontally.
        - 5: Show all channels in the same plot, split vertically.
        - 6: Show all channels in the same plot, split horizontally.

    :param bool anti_alias: Determines whether an antialiased plot is created.
    :param Quantity time_unit: Unit of X-Axis.
    :param progress: Set this parameter to report progress.
    :type progress: :class:`spykeutils.progress_indicator.ProgressIndicator`
    """
    if not spikes or sum((len(l) for l in spikes.itervalues())) < 1:
        raise SpykeException('No spikes for spike waveform plot!')
    if not progress:
        progress = ProgressIndicator()

    progress.begin('Creating waveform plot')
    progress.set_ticks(sum((len(l) for l in spikes.itervalues())))
    win_title = 'Spike waveforms'
    win = PlotDialog(toolbar=True, wintitle=win_title)

    ref_spike = spikes[spikes.keys()[0]][0]
    if ref_spike.waveform is None:
        raise SpykeException(
            'Cannot create waveform plot: At least one spike '
            'has no waveform or sampling rate!')
    ref_units = ref_spike.waveform.units
    channels = range(ref_spike.waveform.shape[1])

    if axes_style <= 2:  # Separate channel plots
        indices = spikes.keys()

        for c in channels:
            pw = BaseCurveWidget(win)
            plot = pw.plot
            plot.set_antialiasing(anti_alias)
            for u in indices:
                color = helper.get_object_color(u)
                for s in spikes[u]:
                    if s.waveform is None or s.sampling_rate is None:
                        raise SpykeException(
                            'Cannot create waveform plot: '
                            'At least one spike has no '
                            'waveform or sampling rate!')
                    x = (sp.arange(s.waveform.shape[0]) /
                         s.sampling_rate).rescale(time_unit)
                    curve = make.curve(
                        x, s.waveform[:, c].rescale(ref_units),
                        title=u.name, color=color)
                    plot.add_item(curve)
                    if axes_style == 1 or c == channels[0]:
                        plot.set_axis_title(BasePlot.Y_LEFT, 'Voltage')
                        plot.set_axis_unit(
                            BasePlot.Y_LEFT, ref_units.dimensionality.string)
                    if axes_style == 2 or c == channels[-1]:
                        plot.set_axis_title(BasePlot.X_BOTTOM, 'Time')
                        plot.set_axis_unit(
                            BasePlot.X_BOTTOM, x.dimensionality.string)
                    progress.step()
            if axes_style == 1:
                win.add_plot_widget(pw, c)
            else:
                win.add_plot_widget(pw, c, 0, c)

        helper.make_window_legend(win, indices, True)

    elif axes_style > 4:  # Only one plot needed
        pw = BaseCurveWidget(win)
        plot = pw.plot
        plot.set_antialiasing(anti_alias)

        if axes_style == 6:  # Horizontal split
            l = _split_plot_hor(channels, spikes, ref_units, time_unit,
                                progress, plot)

            plot.set_axis_title(BasePlot.X_BOTTOM, 'Time')
            plot.set_axis_unit(
                BasePlot.X_BOTTOM, time_unit.dimensionality.string)
        else:  # Vertical split
            channels.reverse()
            max_offset = _find_y_offset(channels, spikes, ref_units)
            l = _split_plot_ver(channels, spikes, ref_units, time_unit,
                                progress, max_offset, plot)

            plot.set_axis_title(BasePlot.Y_LEFT, 'Voltage')
            plot.set_axis_unit(
                BasePlot.Y_LEFT, ref_units.dimensionality.string)

        win.add_plot_widget(pw, 0)
        win.add_legend_option([l], True)

    else:  # One plot per unit
        if axes_style == 3:  # Vertical split
            channels.reverse()

            max_offset = _find_y_offset(channels, spikes, ref_units)

            plot_index = 0
            for u, s in spikes.iteritems():
                pW = BaseCurveWidget(win)
                plot = pW.plot
                plot.set_antialiasing(anti_alias)

                _split_plot_ver(channels, {u: s}, ref_units, time_unit,
                                progress, max_offset, plot)
                if plot_index == 0:
                    plot.set_axis_title(BasePlot.Y_LEFT, 'Voltage')
                    plot.set_axis_unit(
                        BasePlot.Y_LEFT, ref_units.dimensionality.string)
                win.add_plot_widget(pW, plot_index, 0, plot_index)
                plot_index += 1
        else:
            plot_index = 0
            for u, s in spikes.iteritems():
                pW = BaseCurveWidget(win)
                plot = pW.plot
                plot.set_antialiasing(anti_alias)

                _split_plot_hor(channels, {u: s}, ref_units, time_unit,
                                progress, plot)
                win.add_plot_widget(pW, plot_index, plot_index)
                plot_index += 1

            plot.set_axis_title(BasePlot.X_BOTTOM, 'Time')
            plot.set_axis_unit(
                BasePlot.X_BOTTOM, time_unit.dimensionality.string)

    win.add_custom_curve_tools()
    progress.done()
    win.show()

    if axes_style <= 2:
        if len(channels) > 1:
            win.add_x_synchronization_option(True, channels)
            win.add_y_synchronization_option(True, channels)
    elif axes_style <= 4:
        if len(spikes) > 1:
            win.add_x_synchronization_option(True, range(len(spikes)))
            win.add_y_synchronization_option(True, range(len(spikes)))

    return win


def _find_y_offset(channels, spikes, ref_units):
    """ Find y offset needed when plotting spikes split vertically by channel.
    """
    max_y = []
    min_y = []
    for i, c in enumerate(channels):
        max_y.append(
            max(max(s.waveform[:, c].max() for s in d)
                for d in spikes.itervalues()))
        min_y.append(
            min(min(s.waveform[:, c].min() for s in d)
                for d in spikes.itervalues()))

    max_offset = 0 * ref_units
    for i in range(1, len(channels)):
        offset = max_y[i - 1] - min_y[i]
        if offset > max_offset:
            max_offset = offset

    return max_offset


def _split_plot_ver(channels, spikes, ref_units, time_unit, progress,
                    max_offset, plot):
    """ Fill a plot with spikes vertically split by channel. Returns legend.
    """
    legend_items = []
    offset = 0 * ref_units

    for c in channels:
        for u in spikes.keys():
            first_wave = True
            color = helper.get_object_color(u)
            for s in spikes[u]:
                if s.waveform is None or s.sampling_rate is None:
                    raise SpykeException('Cannot create waveform plot: '
                                         'At least one spike has no '
                                         'waveform or sampling rate!')
                x = (sp.arange(s.waveform.shape[0]) /
                     s.sampling_rate).rescale(time_unit)
                curve = make.curve(
                    x,
                    s.waveform[:, c].rescale(ref_units) + offset,
                    u.name, color=color)
                if c == channels[0] and first_wave:
                    legend_curve = make.curve(
                        sp.array([0]), sp.array([0]), u.name,
                        color=color, linewidth=2)
                    legend_items.append(legend_curve)
                    plot.add_item(legend_curve)
                first_wave = False
                plot.add_item(curve)
                progress.step()
        offset += max_offset

    l = make.legend(restrict_items=legend_items)
    plot.add_item(l)
    plot.set_axis_title(BasePlot.X_BOTTOM, 'Time')
    plot.set_axis_unit(BasePlot.X_BOTTOM, time_unit.dimensionality.string)
    return l


def _split_plot_hor(channels, spikes, ref_units, time_unit, progress, plot):
    """ Fill a plot with spikeshorizontally split by channel. Returns legend.
    """
    legend_items = []
    offset = 0 * time_unit

    for c in channels:
        for u in spikes.keys():
            first_wave = True
            color = helper.get_object_color(u)
            for s in spikes[u]:
                if s.waveform is None or s.sampling_rate is None:
                    raise SpykeException(
                        'Cannot create waveform plot: '
                        'At least one spike has no '
                        'waveform or sampling rate!')
                x = (sp.arange(s.waveform.shape[0]) /
                     s.sampling_rate).rescale(time_unit)
                curve = make.curve(
                    x + offset,
                    s.waveform[:, c].rescale(ref_units), u.name,
                    color=color)
                if c == channels[0] and first_wave:
                    legend_curve = make.curve(
                        sp.array([0]), sp.array([0]), u.name,
                        color=color, linewidth=2)
                    legend_items.append(legend_curve)
                    plot.add_item(legend_curve)
                first_wave = False
                plot.add_item(curve)
                progress.step()
        offset += x[-1]
        if c != channels[-1]:
            plot.add_item(
                make.marker((offset, 0), lambda x, y: '',
                            movable=False, markerstyle='|',
                            color='k', linestyle='-', linewidth=1))

    l = make.legend(restrict_items=legend_items)
    plot.add_item(l)
    plot.set_axis_title(BasePlot.Y_LEFT, 'Voltage')
    plot.set_axis_unit(
        BasePlot.Y_LEFT, ref_units.dimensionality.string)

    return l