#! /usr/bin/env python

import sys
import os
import argparse
import inspect
import json
import pickle

from spykeutils.plugin.analysis_plugin import AnalysisPlugin
from spykeutils.plugin.data_provider import DataProvider
from spykeutils.plugin import io_plugin

# Data provider implementations need to be imported so they can be loaded
import spykeutils.plugin.data_provider_stored

try:
    from spykeutils.plot.helper import _needs_qt, ProgressIndicatorDialog
    from PyQt4.QtGui import QApplication
    _needs_qt.app = QApplication([])
    progress = ProgressIndicatorDialog(None)
except ImportError:
    try:  # Use command line progressbar if package is available
        import progressbar

        class ProgressIndicatorConsole(ProgressIndicator):
            """ Implements a progress indicator for the CLI """
            def __init__(self):
                widgets = ['', progressbar.Percentage(), ' ',
                           progressbar.Bar(), ' ', progressbar.ETA()]
                self.bar = progressbar.ProgressBar(widgets=widgets)
                self.tick = 0
                self.maxval = 0

            def set_ticks(self, ticks):
                """ Set the required number of ticks before the operation is
                done """
                self.bar.maxval = ticks
                self.maxval = ticks
                self.tick = 0

            def begin(self, title=''):
                """ Signal that the operation starts """
                self.tick = 0
                self.bar.finished = False
                self.bar.widgets[0] = title + ' '
                self.bar.start()

            def step(self, num_steps=1):
                """ Signal that one or more steps of the operation were
                completed """
                self.tick += num_steps
                if self.tick > self.bar.maxval:
                    self.bar.maxval = self.tick
                self.bar.update(self.tick)
                super(ProgressIndicatorConsole, self).step(num_steps)

            def set_status(self, newStatus):
                """ Set status description """
                self.bar.widgets[0] = newStatus + ' '

            def done(self):
                """ Signal that the operation is done. """
                self.bar.finish()
        progress = ProgressIndicatorConsole()
    except ImportError:
        progress = ProgressIndicator()

def main():
    parser = argparse.ArgumentParser(description='Start an analysis plugin')
    parser.add_argument('Name', type=str, help='Name of analysis class')
    parser.add_argument('Code', type=str, help='Code of the analysis')
    parser.add_argument('Selection', type=str, help='Serialized selection')
    parser.add_argument(
        '-c', '--config', dest='config', type=str,
        help='Pickled configuration of analysis')
    parser.add_argument(
        '-cf', '--codefile', dest='codefile', action='store_const',
        const=True, default=False,
        help='Code represents a filename containing '
             'code (default: Code is a string containing code')
    parser.add_argument(
        '-sf', '--selectionfile', dest='selectionfile',
        action='store_const', const=True, default=False,
        help='Selection represents a filename containing '
             'the serialized selection (default: Selection is a string')
    parser.add_argument(
        '-dd', '--datadir', type=str, help='The data directory')
    parser.add_argument(
        '-io', type=str, default=[], nargs='+', help='IO plugin file paths')

    parsed = parser.parse_known_args()
    args = parsed[0]
    if parsed[1]:
        print >> sys.stderr, ('Warning: the following command options are '
                              'invalid and were ignored:'), parsed[1]

    exc_globals = {}

    if args.codefile:
        execfile(args.Code, exc_globals)
    else:
        exec(args.Code, exc_globals)

    # Load plugin
    plugin = None
    for cl in exc_globals.values():
        if not inspect.isclass(cl):
            continue

        if not issubclass(cl, AnalysisPlugin):
            continue

        if not cl.__name__ == args.Name:
            continue

        plugin = cl()
        break

    # Load IO plugins
    for io in args.io:
        io_plugin.load_from_file(io)

    if not plugin:
        sys.stderr.write('Could not find plugin class, aborting...\n')
        sys.exit(1)

    # Load configuration
    if args.config:
        plugin.set_parameters(pickle.loads(args.config))

    # Load selection
    try:
        if args.selectionfile:
            f = open(args.Selection, 'r')
            sel_string = '\n'.join(f.readlines())
        else:
            sel_string = args.Selection

        sels = json.loads(sel_string)
    except Exception:
        sys.stderr.write('Could not load selection, aborting...\n')
        sys.exit(1)

    selections = []
    for s in sels:
        selections.append(DataProvider.from_data(s))

    if args.datadir and os.path.isdir(args.datadir):
        AnalysisPlugin.data_dir = args.datadir

    selections[0].progress = progress
    plugin.start(selections[0], selections[1:])

if __name__ == '__main__':
    main()