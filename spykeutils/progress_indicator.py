import functools

class CancelException(Exception):
    """ This is raised when a user cancels a progress process. It is used
    by :class:`ProgressIndicator` and its descendants.
    """
    pass


def ignores_cancel(function):
    """ Decorator for functions that should ignore a raised
    :class:`CancelException` and just return nothing in this case
    """
    @functools.wraps(function)
    def inner(*args, **kwargs):
        try:
            return function(*args, **kwargs)
        except CancelException:
            return
    return inner


class ProgressIndicator(object):
    """ Base class for classes indicating progress of a long operation.

    This class does not implement any of the methods and can be used
    as a dummy if no progress indication is needed.
    """

    def set_ticks(self, ticks):
        """ Set the required number of ticks before the operation is done.

        :param int ticks: The number of steps that the operation will take.
        """
        pass

    def begin(self, title=''):
        """ Signal that the operation starts.

        :param string title: The name of the whole operation.
        """
        pass

    def step(self, num_steps=1):
        """ Signal that one or more steps of the operation were completed.

        :param int num_steps: The number of steps that have been completed.
        """
        pass

    def set_status(self, new_status):
        """ Set status description.

        :param string new_status: A description of the current status.
        """
        pass

    def done(self):
        """ Signal that the operation is done. """
        pass