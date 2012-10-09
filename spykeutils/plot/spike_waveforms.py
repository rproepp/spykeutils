"""
.. autofunction:: spikes(spikes, axes_style, anti_alias=False, time_unit=ms, progress=None)
"""

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

       1 Show each channel in a seperate plot.
       2 Show all channels in the same plot vertically.
       3 Show all channels in the same plot horizontally.

    :param bool anti_alias: Determines whether an antialiased plot is created.
    :param Quantity time_unit: The (time) unit for the x axis
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

    indices = spikes.keys()
    ref_spike = spikes[spikes.keys()[0]][0]
    if ref_spike.waveform is None:
        raise SpykeException('Cannot create waveform plot: At least one spike '
                             'has no waveform or sampling rate!')
    ref_units = ref_spike.waveform.units
    channels = range(ref_spike.waveform.shape[1])

    plot = None
    if axes_style == 1: # Separate channel plots
        for c in channels:
            pW = BaseCurveWidget(win)
            plot = pW.plot
            plot.set_antialiasing(anti_alias)
            for u in indices:
                color = helper.get_object_color(u)
                for s in spikes[u]:
                    if s.waveform is None or s.sampling_rate is None:
                        raise SpykeException('Cannot create waveform plot: '
                                             'At least one spike has no '
                                             'waveform or sampling rate!')
                    x = (sp.arange(s.waveform.shape[0]) /
                         s.sampling_rate).rescale(time_unit)
                    curve = make.curve(x, s.waveform[:, c].rescale(ref_units),
                        title=u.name, color=color)
                    plot.add_item(curve)
                    progress.step()
            win.add_plot_widget(pW, c)


        helper.make_window_legend(win, indices, True)
    else: # Only one plot needed
        pW = BaseCurveWidget(win)
        plot = pW.plot
        plot.set_antialiasing(anti_alias)
        legend_items = []

        if axes_style == 3: # Horizontal split
            offset = 0 * time_unit
            for c in channels:
                for u in indices:
                    first_wave = True
                    color = helper.get_object_color(u)
                    for s in spikes[u]:
                        if s.waveform is None or s.sampling_rate is None:
                            raise SpykeException('Cannot create waveform plot: '
                                                 'At least one spike has no '
                                                 'waveform or sampling rate!')
                        x = (sp.arange(s.waveform.shape[0]) /
                             s.sampling_rate).rescale(time_unit)
                        curve = make.curve(x + offset,
                            s.waveform[:, c].rescale(ref_units), u.name,
                            color=color)
                        if c == channels[0] and first_wave == True:
                            legend_items.append(curve)
                        first_wave = False
                        plot.add_item(curve)
                        progress.step()
                offset += x[-1]
                if c != channels[-1]:
                    plot.add_item(make.marker((offset, 0), lambda x,y: '',
                        movable=False, markerstyle='|', color='k',
                        linestyle='-', linewidth=1))
        else: # Vertical split
            channels.reverse()
            # Find plot y offset
            maxY = []
            minY = []
            for i, c in enumerate(channels):
                maxY.append(max(max(s.waveform[:, c].max() for s in d)
                    for d in spikes.itervalues()))
                minY.append(min(min(s.waveform[:, c].min() for s in d)
                    for d in spikes.itervalues()))

            maxOffset = 0 * ref_units
            for i in range(1, len(channels)):
                offset = maxY[i - 1] - minY[i]
                if offset > maxOffset:
                    maxOffset = offset

            offset = 0 * ref_units
            for c in channels:
                for u in indices:
                    first_wave = True
                    color = helper.get_object_color(u)
                    for s in spikes[u]:
                        if s.waveform is None or s.sampling_rate is None:
                            raise SpykeException('Cannot create waveform plot: '
                                                 'At least one spike has no '
                                                 'waveform or sampling rate!')
                        x = (sp.arange(s.waveform.shape[0]) /
                             s.sampling_rate).rescale(time_unit)
                        curve = make.curve(x,
                            s.waveform[:, c].rescale(ref_units) + offset,
                            u.name, color=color)
                        if c == channels[0] and first_wave == True:
                            legend_items.append(curve)
                        first_wave = False
                        plot.add_item(curve)
                        progress.step()
                offset += maxOffset

        l = make.legend(restrict_items=legend_items)
        plot.add_item(l)
        win.add_plot_widget(pW, 0)
        win.add_legend_option([l], True)

    win.add_custom_curve_tools()
    progress.done()
    win.show()

    plot.set_axis_title(BasePlot.X_BOTTOM, 'Time')
    plot.set_axis_unit(BasePlot.X_BOTTOM, x.dimensionality.string)

    if axes_style == 1:
        win.add_x_synchronization_option(True, channels)
        win.add_y_synchronization_option(True, channels)

    if len(channels) == 1 or axes_style > 1:
        plot.set_axis_title(BasePlot.Y_LEFT, 'Voltage')
        plot.set_axis_unit(BasePlot.Y_LEFT, ref_units.dimensionality.string)