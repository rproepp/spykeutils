import functools

from PyQt4.QtGui import QApplication
from guiqwt.builder import make

import neo

class _MarkerName:
    """ Helper class to create marker name functions for different strings.
    """
    def __init__(self, name):
        self.name = name

    #noinspection PyUnusedLocal
    def get_name(self, x, y):
        return self.name


def needs_qt(function):
    """ Decorator for functions making sure that an initialized PyQt exists.
    """
    @functools.wraps(function)
    def inner(*args, **kwargs):
        app = None
        if not QApplication.instance():
            app = QApplication([])
        function(*args, **kwargs)
        if app:
            app.exec_()
    return inner

def create_unit(name, id):
    u = neo.Unit(name)
    u.annotate(unique_id=id)
    return u

_color = ['#00ffff', # Cyan
    '#ff00ff', # Magenta
   '#ff0000', # Red
   '#00ff00', # Green
   '#0000ff', # Blue
   '#cccc00', # Dark yellow
   '#000077', # Navy
   '#770000', # Maroon
   '#007777', # Teal
   '#777700', # Olive
   '#770077', # Purple
   '#a146ff'] # Violet

def get_color(entity_id):
    """ Return a color for an int.
    """
    return _color[entity_id % len(_color)]

def get_unit_color(unit):
    """ Return a color for a Neo unit object, based on the 'unique_id'
        annotation.
    """
    if 'unique_id' in unit.annotations:
        return get_color(unit.annotations['unique_id'])
    return get_color(hash(unit))

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