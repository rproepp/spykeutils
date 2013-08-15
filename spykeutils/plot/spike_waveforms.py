import scipy as sp
import quantities as pq

from guiqwt.builder import make
from guiqwt.baseplot import BasePlot
from guiqwt.plot import BaseCurveWidget
from PyQt4 import Qt

from ..progress_indicator import ProgressIndicator
from .. import SpykeException
from dialog import PlotDialog
import helper


@helper.needs_qt
def spikes(spikes, axes_style, strong=None, anti_alias=False, fade=1.0,
           subplot_layout=0, time_unit=pq.ms, progress=None):
    """ Create a plot dialog with spike waveforms. Assumes that all spikes
    have waveforms with the same number of channels.

    :param dict spikes: A dictionary of :class:`neo.core.Spike` lists.
    :param int axes_style: Plotting mode. The following values are possible:

        - 1: Show each channel in a seperate plot, split vertically.
        - 2: Show each channel in a separate plot, split horizontally.
        - 3: Show each key of ``spikes`` in a separate plot,
          channels are split vertically.
        - 4: Show each key of ``spikes`` in a separate plot,
          channels are split horizontally.
        - 5: Show all channels in the same plot, split vertically.
        - 6: Show all channels in the same plot, split horizontally.

    :param dict strong: A dictionary of :class:`neo.core.Spike` lists. When
        given, these spikes are shown as thick lines on top of the regular
        spikes in the respective plots.
    :param bool anti_alias: Determines whether an antialiased plot is created.
    :param float fade: Vary transparency by segment. For values > 0, the first
        spike for each unit is displayed with the corresponding alpha
        value and alpha is linearly interpolated until it is 1 for the
        last spike. For values < 0, alpha is 1 for the first spike and
        ``fade`` for the last spike. Does not affect spikes from ``strong``.
    :param bool subplot_layout: The way subplots are arranged on the window:

        - 0: Linear - horizontally or vertically,
          depending on ``axis_style``.
        - 1: Square - this layout tries to have the same number of plots per
          row and per column.

    :param Quantity time_unit: Unit of X-Axis.
    :param progress: Set this parameter to report progress.
    :type progress: :class:`spykeutils.progress_indicator.ProgressIndicator`
    """
    if (not spikes or sum((len(l) for l in spikes.itervalues())) < 1) and \
            (not strong or sum((len(l) for l in strong.itervalues())) < 1):
        raise SpykeException('No spikes for spike waveform plot!')
    if not progress:
        progress = ProgressIndicator()
    if strong is None:
        strong = {}

    progress.begin('Creating waveform plot')
    progress.set_ticks(
        sum((len(l) for l in spikes.itervalues())) +
        sum((len(l) for l in strong.itervalues())))
    win_title = 'Spike waveforms'
    win = PlotDialog(toolbar=True, wintitle=win_title)

    try:
        ref_spike = spikes[spikes.keys()[0]][0]
    except IndexError:
        ref_spike = strong[strong.keys()[0]][0]
    if ref_spike.waveform is None:
        raise SpykeException(
            'Cannot create waveform plot: At least one spike '
            'has no waveform or sampling rate!')
    ref_units = ref_spike.waveform.units
    channels = range(ref_spike.waveform.shape[1])

    # Keys from spikes and strong without duplicates in original order
    seen = set()
    indices = [k for k in spikes.keys() + strong.keys()
               if k not in seen and not seen.add(k)]

    if axes_style <= 2:  # Separate channel plots
        for c in channels:
            pw = BaseCurveWidget(win)
            plot = pw.plot
            plot.set_antialiasing(anti_alias)
            for u in spikes:
                color = helper.get_object_color(u)
                qcol = Qt.QColor(color)
                alpha = fade if fade > 0.0 else 1.0
                alpha_step = 1.0 - fade if fade > 0.0 else -1.0 - fade
                alpha_step /= len(spikes[u])
                if len(spikes[u]) == 1:
                    alpha = 1.0

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

                    qcol.setAlphaF(alpha)
                    curve.setPen(Qt.QPen(qcol))
                    alpha += alpha_step

                    plot.add_item(curve)
                    progress.step()

            for u in strong:
                color = helper.get_object_color(u)
                for s in strong[u]:
                    x = (sp.arange(s.waveform.shape[0]) /
                         s.sampling_rate).rescale(time_unit)
                    outline = make.curve(
                        x, s.waveform[:, c].rescale(ref_units),
                        color='#000000', linewidth=4)
                    curve = make.curve(
                        x, s.waveform[:, c].rescale(ref_units),
                        color=color, linewidth=2)
                    plot.add_item(outline)
                    plot.add_item(curve)
                    progress.step()

            _add_plot(plot, pw, win, c, len(channels), subplot_layout,
                      axes_style, time_unit, ref_units)

        helper.make_window_legend(win, indices, True)
    elif axes_style > 4:  # Only one plot needed
        pw = BaseCurveWidget(win)
        plot = pw.plot
        plot.set_antialiasing(anti_alias)

        if axes_style == 6:  # Horizontal split
            l = _split_plot_hor(channels, spikes, strong, fade, ref_units,
                                time_unit, progress, plot)

            plot.set_axis_title(BasePlot.X_BOTTOM, 'Time')
            plot.set_axis_unit(
                BasePlot.X_BOTTOM, time_unit.dimensionality.string)
        else:  # Vertical split
            channels.reverse()

            max_offset = _find_y_offset(channels, spikes, strong, ref_units)
            l = _split_plot_ver(channels, spikes, strong, fade, ref_units,
                                time_unit, progress, max_offset, plot)

            plot.set_axis_title(BasePlot.Y_LEFT, 'Voltage')
            plot.set_axis_unit(
                BasePlot.Y_LEFT, ref_units.dimensionality.string)

        win.add_plot_widget(pw, 0)
        win.add_legend_option([l], True)
    else:  # One plot per unit
        if axes_style == 3:
            channels.reverse()
            max_offset = _find_y_offset(channels, spikes, strong, ref_units)

        for i, u in enumerate(indices):
            pw = BaseCurveWidget(win)
            plot = pw.plot
            plot.set_antialiasing(anti_alias)

            spk = {}
            if u in spikes:
                spk[u] = spikes[u]
            st = {}
            if u in strong:
                st[u] = strong[u]

            if axes_style == 3:  # Vertical split
                _split_plot_ver(channels, spk, st, fade, ref_units,
                                time_unit, progress, max_offset, plot)
            else:  # Horizontal split
                _split_plot_hor(channels, spk, st, fade, ref_units,
                                time_unit, progress, plot)

            _add_plot(plot, pw, win, i, len(indices), subplot_layout,
                      axes_style, time_unit, ref_units)

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


def _add_plot(plot, widget, win, index, total, subplot_layout, axes_style,
              time_unit, ref_units):
    """ Add a plot to the window in the right position with the correct axis
    labels.
    """
    if subplot_layout == 0:
        if axes_style == 1 or axes_style == 4 or index == 0:
            plot.set_axis_title(BasePlot.Y_LEFT, 'Voltage')
            plot.set_axis_unit(
                BasePlot.Y_LEFT, ref_units.dimensionality.string)
        if axes_style == 2 or axes_style == 3 or index == total - 1:
            plot.set_axis_title(BasePlot.X_BOTTOM, 'Time')
            plot.set_axis_unit(
                BasePlot.X_BOTTOM, time_unit.dimensionality.string)
        if axes_style == 1 or axes_style == 4:
            win.add_plot_widget(widget, index)
        else:
            win.add_plot_widget(widget, index, 0, index)
    else:
        size = int(sp.sqrt(total) + 0.99)
        if index % size == 0:
            plot.set_axis_title(BasePlot.Y_LEFT, 'Voltage')
            plot.set_axis_unit(
                BasePlot.Y_LEFT, ref_units.dimensionality.string)
        if index >= total - size:
            plot.set_axis_title(BasePlot.X_BOTTOM, 'Time')
            plot.set_axis_unit(
                BasePlot.X_BOTTOM, time_unit.dimensionality.string)
        win.add_plot_widget(widget, index, index / size, index % size)


def _find_y_offset(channels, spikes, strong, ref_units):
    """ Find y offset needed when plotting spikes split vertically by channel.
    """
    max_y = []
    min_y = []
    if spikes:
        for i, c in enumerate(channels):
            max_y.append(
                max(max(s.waveform[:, c].max() for s in d)
                    for d in spikes.itervalues()))
            min_y.append(
                min(min(s.waveform[:, c].min() for s in d)
                    for d in spikes.itervalues()))

    if strong:
        for i, c in enumerate(channels):
            max_y.append(
                max(max(s.waveform[:, c].max() for s in d)
                    for d in strong.itervalues()))
            min_y.append(
                min(min(s.waveform[:, c].min() for s in d)
                    for d in strong.itervalues()))

    max_offset = 0 * ref_units
    for i in range(1, len(channels)):
        offset = max_y[i - 1] - min_y[i]
        if offset > max_offset:
            max_offset = offset

    return max_offset


def _split_plot_ver(channels, spikes, strong, fade, ref_units, time_unit,
                    progress, max_offset, plot):
    """ Fill a plot with spikes vertically split by channel. Returns legend.
    """
    offset = 0 * ref_units

    for c in channels:
        for u in spikes:
            color = helper.get_object_color(u)
            qcol = Qt.QColor(color)
            alpha = fade if fade > 0.0 else 1.0
            alpha_step = 1.0 - fade if fade > 0.0 else -1.0 - fade
            alpha_step /= len(spikes[u])
            if len(spikes[u]) == 1:
                alpha = 1.0

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

                qcol.setAlphaF(alpha)
                curve.setPen(Qt.QPen(qcol))
                alpha += alpha_step

                plot.add_item(curve)
                progress.step()

        for u in strong:
            color = helper.get_object_color(u)
            for s in strong[u]:
                x = (sp.arange(s.waveform.shape[0]) /
                     s.sampling_rate).rescale(time_unit)
                outline = make.curve(
                    x, s.waveform[:, c].rescale(ref_units) + offset,
                    color='#000000', linewidth=4)
                curve = make.curve(
                    x, s.waveform[:, c].rescale(ref_units) + offset,
                    color=color, linewidth=2)
                plot.add_item(outline)
                plot.add_item(curve)
                progress.step()

        offset += max_offset

    l = _add_legend(plot, spikes, strong)

    return l


def _split_plot_hor(channels, spikes, strong, fade, ref_units, time_unit,
                    progress, plot):
    """ Fill a plot with spikeshorizontally split by channel. Returns legend.
    """
    offset = 0 * time_unit

    for c in channels:
        x_off = 0 * time_unit
        for u in spikes:
            color = helper.get_object_color(u)
            qcol = Qt.QColor(color)
            alpha = fade if fade > 0.0 else 1.0
            alpha_step = 1.0 - fade if fade > 0.0 else -1.0 - fade
            alpha_step /= len(spikes[u])
            if len(spikes[u]) == 1:
                alpha = 1.0

            for s in spikes[u]:
                if s.waveform is None or s.sampling_rate is None:
                    raise SpykeException(
                        'Cannot create waveform plot: '
                        'At least one spike has no '
                        'waveform or sampling rate!')
                x = (sp.arange(s.waveform.shape[0]) /
                     s.sampling_rate).rescale(time_unit)
                x_off = max(x_off, x[-1])
                curve = make.curve(
                    x + offset,
                    s.waveform[:, c].rescale(ref_units), u.name,
                    color=color)

                qcol.setAlphaF(alpha)
                curve.setPen(Qt.QPen(qcol))
                alpha += alpha_step

                plot.add_item(curve)
                progress.step()

        for u in strong:
            color = helper.get_object_color(u)
            for s in strong[u]:
                x = (sp.arange(s.waveform.shape[0]) /
                     s.sampling_rate).rescale(time_unit)
                x_off = max(x_off, x[-1])
                outline = make.curve(
                    x + offset, s.waveform[:, c].rescale(ref_units),
                    color='#000000', linewidth=4)
                curve = make.curve(
                    x + offset, s.waveform[:, c].rescale(ref_units),
                    color=color, linewidth=2)
                plot.add_item(outline)
                plot.add_item(curve)
                progress.step()

        offset += x_off
        if c != channels[-1]:
            plot.add_item(
                make.marker((offset, 0), lambda x, y: '',
                            movable=False, markerstyle='|',
                            color='k', linestyle='-', linewidth=1))

    l = _add_legend(plot, spikes, strong)

    return l


def _add_legend(plot, spikes, strong):
    # Keys from spikes and strong without duplicates in original order
    seen = set()
    indices = [k for k in spikes.keys() + strong.keys()
               if k not in seen and not seen.add(k)]
    legend_items = []

    for u in indices:
        legend_curve = make.curve(
            sp.array([0]), sp.array([0]), u.name,
            color=helper.get_object_color(u), linewidth=2)
        legend_items.append(legend_curve)
        plot.add_item(legend_curve)

    l = make.legend(restrict_items=legend_items)
    plot.add_item(l)

    return l