""" Functions for functional programming. """


def apply_to_dict(fn, dictionary, *args):
    """ Applies a function to all spike trains in a dictionary of spike train
    lists.

    :param func fn: Function to apply. Should take a spike train as first
        argument.
    :param dict dictionary: Dictionary of spike train lists to apply the
        function to.
    :param args: Additional arguments which will be passed to `fn`.
    :returns: A new dictionary with the same keys as `dictionary`.
    :rtype: dict
    """

    applied = {}
    for k in dictionary:
        applied[k] = [fn(st, *args) for st in dictionary[k]]
    return applied
