
import neo
import quantities as pq
import scipy as sp
import _scipy_quantities as spq


def apply_to_dict(fn, dictionary, *args):
    """ Applies a function to all spike trains in a dictionary of spike train
    sequences.

    :param function fn: Function to apply. Should take a
        :class:`neo.core.SpikeTrain` as first argument.
    :param dict dictionary: Dictionary of sequences of
        :class:`neo.core.SpikeTrain` objects to apply the function to.
    :param args: Additional arguments which will be passed to `fn`.
    :returns: A new dictionary with the same keys as `dictionary`.
    :rtype: dict
    """

    applied = {}
    for k in dictionary:
        applied[k] = [fn(st, *args) for st in dictionary[k]]
    return applied


def bin_spike_trains(trains, sampling_rate, t_start=None, t_stop=None):
    """ Creates a binned representation of a spike train.

    :param dict trains: A dictionary of sequences of
        :class:`neo.core.SpikeTrain` objects.
    :param sampling_rate: The sampling rate which will be used to bin
        the spike trains as inverse time scalar.
    :type sampling_rate: Quantity scalar
    :type t_start: The desired time for the start of the first bin as time
        scalar. It will be the minimum start time of all spike trains if `None`
        is passed.
    :type t_start: Quantity scalar
    :param t_stop: The desired time for the end of the last bin as time scalar.
        It will be the maximum stop time of all spike trains if `None` is
        passed.
    :type t_stop: Quantity scalar
    :returns: A dictionary (with the same indices as ``trains``) of lists
        of spike train counts and the bin borders.
    :rtype: dict, Quantity 1D with time units
    """

    if t_start is None or t_stop is None:
        max_start, max_stop = maximum_spike_train_interval(trains)
        if t_start is None:
            t_start = max_start
        if t_stop is None:
            t_stop = max_stop

    t_start = t_start.rescale(t_stop.units)

    duration = t_stop - t_start
    num_bins = (sampling_rate * duration).simplified + 1
    bins = spq.linspace(t_start.magnitude, t_stop.magnitude, num_bins) * \
        t_stop.units
    return apply_to_dict(_bin_single_spike_train, trains, bins), bins


def _bin_single_spike_train(train, bins):
    """ Return a binned representation of SpikeTrain object.

    :param train: A spike train to bin.
    :type train: :class:`neo.core.SpikeTrain`
    :param bins: The bin edges, including the rightmost edge, with time units.
    :type bins: Quantity 1D
    :returns: The binned spike train.
    :rtype: 1-D array
    """
    return sp.histogram(train.rescale(bins.units), bins)[0]


def st_concatenate(trains):
    """ Concatenates spike trains.

    :param sequence trains: :class:`neo.core.SpikeTrain` objects to concatenate.
    :returns: A spike train consisting of the concatenated spike trains. The
        spikes will be in the order of the given spike trains and `t_start` and
        `t_stop` will be set to the minimum and maximum value.
    :rtype: :class:`neo.core.SpikeTrain`
    """

    t_start, t_stop = maximum_spike_train_interval({0: trains})
    return neo.SpikeTrain(
        spq.concatenate([train.view(type=pq.Quantity) for train in trains]),
        t_start=t_start, t_stop=t_stop)


def minimum_spike_train_interval(
        trains, t_start=-sp.inf * pq.s, t_stop=sp.inf * pq.s):
    """ Computes the maximum starting time and minimum end time that all
    given spike trains share. This yields the shortest interval shared by all
    spike trains.

    :param dict trains: A dictionary of sequences of
        :class:`neo.core.SpikeTrain` objects.
    :param t_start: Minimal starting time to return as time scalar.
    :param t_stop: Maximum end time to return as time scalar.
    :returns: Maximum shared t_start time and minimum shared t_stop time as time
        scalars.
    :rtype: Quantity scalar, Quantity scalar
    """
    # Load data and find shortest spike train
    for st in trains.itervalues():
        if len(st) > 0:
            # Minimum length of spike of all spike trains for this unit
            t_start = max(t_start, max((t.t_start for t in st)))
            t_stop = min(t_stop, min((t.t_stop for t in st)))

    return t_start, t_stop


def maximum_spike_train_interval(
        trains, t_start=sp.inf * pq.s, t_stop=-sp.inf * pq.s):
    """ Computes the minimum starting time and maximum end time of all
    given spike trains. This yields an interval containing the spikes of
    all spike trains.

    :param dict trains: A dictionary of sequences of
        :class:`neo.core.SpikeTrain` objects.
    :param t_start: Maximum starting time to return as time scalar.
    :param t_stop: Minimum end time to return as time scalar.
    :returns: Minimum t_start time and maximum t_stop time as time scalars.
    :rtype: Quantity scalar, Quantity scalar
    """
    for st in trains.itervalues():
        if len(st) > 0:
            t_start = min(t_start, min((t.t_start for t in st)))
            t_stop = max(t_stop, max((t.t_stop for t in st)))

    return t_start, t_stop
