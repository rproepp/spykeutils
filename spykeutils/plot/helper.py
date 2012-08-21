import functools

import scipy as sp

from PyQt4.QtGui import QApplication
from guiqwt.builder import make

class _MarkerName:
    """ Helper class to create marker name functions for different strings.
    """
    def __init__(self, name):
        self.name = name

    #noinspection PyUnusedLocal
    def get_name(self, x, y):
        return self.name


def _needs_qt(function):
    @functools.wraps(function)
    def inner(*args, **kwargs):
        app = None
        if not QApplication.instance():
            app = QApplication([])
        function(*args, **kwargs)
        if app:
            app.exec_()
    return inner

# Make need_qt decorator preserve signature if decorator package is available
try:
    from decorator import FunctionMaker

    def decorator_apply(dec, func):
        return FunctionMaker.create(
            func, 'return decorated(%(signature)s)',
            dict(decorated=dec(func)), __wrapped__=func)

    def needs_qt(func):
        """ Decorator for functions making sure that an initialized PyQt exists.
        """
        return decorator_apply(_needs_qt, func)
except ImportError:
    needs_qt = _needs_qt


# Optimum contrast color palette (without white and black), see
# http://web.media.mit.edu/~wad/color/palette.html
__default_colors = ['#575757', # Dark Gray
    '#ad2323', # Red
   '#2a4bd7', # Blue
   '#296914', # Green
   '#614a19', # Brown (lower R to better distinguish from purple)
   '#8126c0', # Purple
   '#a0a0a0', # Light Gray
   '#81c57a', # Light Green
   '#9dafff', # Light Blue
   '#29d0d0', # Cyan
   '#ff9233', # Orange
   '#ffee33', # Yellow
   '#b6ab88', # Tan (darkened for better visibility on white background)
   '#ff89d1'] # Pink (darkened for better visibility on white background)

__colors = __default_colors

def get_color(entity_id):
    """ Return a color for an int.
    """
    return __colors[entity_id % len(__colors)]

def get_object_color(unit):
    """ Return a color for a Neo object, based on the 'unique_id'
    annotation. If the annotation is not present, return a color based
    on the hash of the object.
    """
    try:
        if 'unique_id' in unit.annotations:
            return get_color(unit.annotations['unique_id'])
    except Exception:
        return get_color(hash(unit))
    return get_color(hash(unit))

def set_color_scheme(colors):
    """ Set the color scheme used in plots.

    :param sequence colors: A list of strings with HTML-style color codes
        (e.g. ``'#ffffff'`` for white). If this is ``None`` or empty,
        the default color scheme will be set.
    """
    global __colors
    global __default_colors
    if not colors:
        __colors = __default_colors
    else:
        __colors = colors


def add_events(plot, events, units=None):
    """ Add Event markers to a guiqwt plot.

    :param plot: The plot object.
    :type plot: :class:`guiqwt.baseplot.BasePlot`
    :param sequence events: The events (neo :class:`neo.Event` objects).
    :param Quantity units: The x-scale of the plot. If this is ``None``,
        the time unit of the events will be use.
    """
    for m in events:
        nameObject = _MarkerName(m.label)
        if units:
            time = m.time.rescale(units)
        else:
            time = m.time
        plot.add_item(make.marker((time, 0), nameObject.get_name,
            movable=False, markerstyle='|', color='k', linestyle=':',
            linewidth=1))

def add_spikes(plot, train, color='k', spike_width=1, spike_height=20000,
               y_offset = 0, name='', units=None):
    """ Add all spikes from a spike train to a guiqwt plot as vertical lines.

    :param plot: The plot object.
    :type plot: :class:`guiqwt.baseplot.BasePlot`
    :param train: A spike train with the spike times to show.
    :type train: :class:`neo.core.SpikeTrain`
    :param str color: The color for the spikes.
    :param int spike_width: The width of the shown spikes in pixels.
    :param int spike_height: The height of the shown spikes in pixels.
    :param float y_offset: An offset for the drawing position on the y-axis.
    :param str name: The name of the spike train.
    :param Quantity units: The x-scale of the plot. If this is ``None``,
        the time unit of the events will be use.
    :returns: The plot item added for the spike train
    """
    if units:
        train = train.rescale(units)

    spikes = make.curve(train, sp.zeros(len(train)) + y_offset,
        name, 'k', 'NoPen', linewidth=0, marker='Rect',
        markerfacecolor=color, markeredgecolor=color)

    s = spikes.symbol()
    s.setSize(spike_width-1, spike_height)
    spikes.setSymbol(s)
    plot.add_item(spikes)

    return spikes

def add_epochs(plot, epochs, units=None):
    """ Add Epoch markers to a guiqwt plot.

    :param plot: The plot object.
    :type plot: :class:`guiqwt.baseplot.BasePlot`
    :param sequence epochs: The epochs (neo :class:`neo.Epoch` objects).
    :param units: The x-scale of the plot. If this is ``None``,
        the time unit of the events will be use.
    """
    for e in epochs:
        if units:
            start = e.time.rescale(units)
            end = (e.time + e.duration).rescale(units)
        else:
            start = e.time
            end = e.time + e.duration
        time = (start + end) / 2.0

        o = make.range(start, end)
        o.setTitle(e.label)
        o.set_readonly(True)
        o.set_movable(False)
        o.set_resizable(False)
        o.set_selectable(False)
        o.set_rotatable(False)
        plot.add_item(o)

        nameObject = _MarkerName(e.label)
        plot.add_item(make.marker((time, 0), nameObject.get_name,
            movable=False, markerstyle='|', color='k', linestyle='NoPen',
            linewidth=1))

def make_window_legend(win, units, show_option=None):
    """ Create a legend in a PlotDialog for a given sequence of neo objects.

    :param win: The window where the legend will be added.
    :type win: :class:`spykeutils.plot.dialogs.PlotDialog`
    :param sequence units: A list of neo objects which will be included in
        the legend.
    :param bool show_option: Determines whether a toggle for the legend
        will be shown (if the parameter is not ``None``) and if the legend
        is visible initially (``True``/``False``).
    """
    if not units:
        return

    legend = []
    for u in units:
        legend.append((get_object_color(u), u.name))
    win.add_color_legend(legend, show_option)