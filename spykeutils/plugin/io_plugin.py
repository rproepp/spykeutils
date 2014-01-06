import traceback
import inspect
import neo
from neo.io.baseio import BaseIO

from .. import SpykeException


def load_from_file(path):
    """ Load IO plugins from a Python file. Inserts
    the loaded plugins into the neo.iolist.

    :param str path: The path to the file to search for IO plugins.
    """
    f = open(path)
    load_from_string(f.read(), path)


def load_from_string(code, path='<string>'):
    """ Load IO plugins from Python code. Inserts
    the loaded plugins into the neo.iolist.

    :param str code: The IO plugin code.
    :param str path: The path for the IO plugin.
    """
    exc_globals = {}
    try:
        exec(code, exc_globals)
    except Exception:
        raise SpykeException('Error during execution of ' +
                             'potential Neo IO file ' + path + ':\n' +
                             traceback.format_exc() + '\n')

    for cl in exc_globals.values():
        if not inspect.isclass(cl):
            continue

        # Should be a subclass of BaseIO...
        if not issubclass(cl, BaseIO):
            continue
            # but should not be BaseIO (can happen when directly imported)
        if cl == BaseIO:
            continue

        cl._is_spyke_plugin = True
        cl._python_file = path
        neo.io.iolist.insert(0, cl)