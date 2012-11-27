""" Additional guiqwt tools to facilitate plot navigation.
"""

from PyQt4 import QtGui
from PyQt4.QtCore import Qt
from PyQt4.QtGui import QMessageBox
try:
    from PyQt4 import QtCore
    _fromUtf8 = QtCore.QString.fromUtf8
except AttributeError:
    _fromUtf8 = lambda s: s

from guiqwt.tools import CommandTool, DefaultToolbarID, InteractiveTool
from guidata.qthelpers import get_std_icon
from guiqwt.config import _
from guiqwt.events import (setup_standard_tool_filter, PanHandler)

import icons_rc


class HomeTool(CommandTool):
    """ A command to show all elements in the plot (same as pressing the
    middle mouse button).
    """

    def __init__(self, manager, toolbar_id=DefaultToolbarID):
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(_fromUtf8(':/Plottools/Home')),
            QtGui.QIcon.Normal, QtGui.QIcon.Off)

        super(HomeTool, self).__init__(manager, 'Home', icon,
            toolbar_id=toolbar_id)

    def activate_command(self, plot, checked):
        """ Activate that command!
        """
        plot.do_autoscale()


class PanTool(InteractiveTool):
    """ Allows panning with the left mouse button.
    """
    TITLE = _("Pan")
    ICON = "move.png"
    CURSOR = Qt.OpenHandCursor

    def setup_filter(self, baseplot):
        filter = baseplot.filter
        start_state = filter.new_state()
        PanHandler(filter, Qt.LeftButton, start_state=start_state)

        return setup_standard_tool_filter(filter, start_state)


class HelpTool(CommandTool):
    """ A help tool that includes a message what a single middle click
    does, otherwise identical to the guiqwt tool with the same name.
    """
    def __init__(self, manager, toolbar_id=DefaultToolbarID):
        super(HelpTool,self).__init__(manager, "Help",
            get_std_icon("DialogHelpButton", 16),
            toolbar_id=toolbar_id)

    def activate_command(self, plot, checked):
        """Activate tool"""
        QMessageBox.information(plot, "Help",
            """Keyboard/mouse shortcuts:
  - single left-click: item (curve, image, ...) selection
  - single right-click: context-menu relative to selected item
  - single middle click: home
  - shift: on-active-curve (or image) cursor
  - alt: free cursor
  - left-click + mouse move: move item (when available)
  - middle-click + mouse move: pan
  - right-click + mouse move: zoom""")