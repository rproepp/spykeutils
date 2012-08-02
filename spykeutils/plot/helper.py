import functools

from PyQt4.QtGui import QApplication
from guiqwt.builder import make

import neo

class MarkerName:
    """ Helper class to create marker name functions for different strings
    """
    def __init__(self, name):
        self.name = name

    #noinspection PyUnusedLocal
    def get_name(self, x, y):
        return self.name


def needs_qt(function):
    """ Decorator for functions making sure that an initialized PyQt exists
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
    """ Return a color for an int
    """
    return _color[entity_id % len(_color)]

def get_unit_color(unit):
    """ Return a color for a Neo unit object, based on the 'unique_id'
        annotation
    """
    if 'unique_id' in unit.annotations:
        return get_color(unit.annotations['unique_id'])
    return get_color(hash(unit))

def add_markers(plot, markers):
    """ Add markers to a plot
        :param plot: The plot object
        :param sequence markers: The markers (neo `Event` objects)
    """
    for m in markers:
        nameObject = MarkerName(m.label)
        plot.add_item(make.marker((m.time, 0), nameObject.get_name,
            movable=False, markerstyle='|', color='k', linestyle=':',
            linewidth=1))