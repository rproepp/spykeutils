import neo
try:
    import neo.description
    HAS_DESCRIPTION = True
except ImportError:
    HAS_DESCRIPTION = False
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
    :param args: Additional arguments which will be passed to ``fn``.
    :returns: A new dictionary with the same keys as ``dictionary``.
    :rtype: dict
    """

    applied = {}
    for k in dictionary:
        applied[k] = [fn(st, *args) for st in dictionary[k]]
    return applied


def bin_spike_trains(trains, sampling_rate, t_start=None, t_stop=None):
    """ Creates binned representations of spike trains.

    :param dict trains: A dictionary of sequences of
        :class:`neo.core.SpikeTrain` objects.
    :param sampling_rate: The sampling rate which will be used to bin
        the spike trains as inverse time scalar.
    :type sampling_rate: Quantity scalar
    :type t_start: The desired time for the start of the first bin as time
        scalar. It will be the minimum start time of all spike trains if
        ``None`` is passed.
    :type t_start: Quantity scalar
    :param t_stop: The desired time for the end of the last bin as time scalar.
        It will be the maximum stop time of all spike trains if ``None`` is
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
    num_bins = (sampling_rate * duration).simplified
    bins = sp.arange(num_bins + 1) * (duration / num_bins) + t_start
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


def concatenate_spike_trains(trains):
    """ Concatenates spike trains.

    :param sequence trains: :class:`neo.core.SpikeTrain` objects to
        concatenate.
    :returns: A spike train consisting of the concatenated spike trains. The
        spikes will be in the order of the given spike trains and ``t_start``
        and ``t_stop`` will be set to the minimum and maximum value.
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
    :param t_start: Minimal starting time to return.
    :type t_start: Quantity scalar
    :param t_stop: Maximum end time to return. If ``None``, infinity is used.
    :type t_stop: Quantity scalar
    :returns: Maximum shared t_start time and minimum shared t_stop time as
        time scalars.
    :rtype: Quantity scalar, Quantity scalar
    """
    if t_stop is None:
        t_stop = sp.inf * pq.s

    # Load data and find shortest spike train
    for st in trains.itervalues():
        if len(st) > 0:
            # Minimum length of spike of all spike trains for this unit
            t_start = max(t_start, max((t.t_start for t in st)))
            t_stop = min(t_stop, min((t.t_stop for t in st)))

    if t_stop == sp.inf * pq.s:
        t_stop = t_start

    return t_start, t_stop


def maximum_spike_train_interval(
        trains, t_start=sp.inf * pq.s, t_stop=-sp.inf * pq.s):
    """ Computes the minimum starting time and maximum end time of all
    given spike trains. This yields an interval containing the spikes of
    all spike trains.

    :param dict trains: A dictionary of sequences of
        :class:`neo.core.SpikeTrain` objects.
    :param t_start: Maximum starting time to return.
    :type t_start: Quantity scalar
    :param t_stop: Minimum end time to return. If ``None``, infinity is used.
    :type t_stop: Quantity scalar
    :returns: Minimum t_start time and maximum t_stop time as time scalars.
    :rtype: Quantity scalar, Quantity scalar
    """
    if t_stop is None:
        t_stop = sp.inf * pq.s

    for st in trains.itervalues():
        if len(st) > 0:
            t_start = min(t_start, min((t.t_start for t in st)))
            t_stop = max(t_stop, max((t.t_stop for t in st)))

    return t_start, t_stop


def _handle_orphans(obj, remove):
    """ Removes half-orphaned Spikes and SpikeTrains that occur when
    removing an object upwards in the hierarchy.
    """
    if isinstance(obj, neo.Segment):
        for s in obj.spikes:
            if s.unit:
                if not remove:
                    s.segment = None
                else:
                    try:
                        s.unit.spikes.remove(s)
                    except ValueError:
                        pass

        for st in obj.spiketrains:
            if st.unit:
                if not remove:
                    st.segment = None
                else:
                    try:
                        st.unit.spiketrains.remove(st)
                    except ValueError:
                        pass
    elif isinstance(obj, neo.Unit):
        for s in obj.spikes:
            if s.segment:
                if not remove:
                    s.unit = None
                else:
                    try:
                        s.segment.spikes.remove(s)
                    except ValueError:
                        pass

        for st in obj.spiketrains:
            if st.segment:
                if not remove:
                    st.unit = None
                else:
                    try:
                        st.segment.spiketrains.remove(st)
                    except ValueError:
                        pass
    elif isinstance(obj, neo.RecordingChannelGroup):
        for u in obj.units:
            _handle_orphans(u, remove)


def remove_from_hierarchy(obj, remove_half_orphans=True):
    """ Removes a Neo object from the hierarchy it is embedded in. Mostly
    downward links are removed (except for possible links in
    :class:`neo.core.Spike` or :class:`neo.core.SpikeTrain` objects).
    For example, when ``obj`` is a :class:`neo.core.Segment`, the link from
    its parent :class:`neo.core.Block` will be severed. Also, all links to
    the segment from its spikes and spike trains will be severed.

    :param obj: The object to be removed.
    :type obj: Neo object
    :param bool remove_half_orphans: When True, :class:`neo.core.Spike`
        and :class:`neo.core.SpikeTrain` belonging to a
        :class:`neo.core.Segment` or :class:`neo.core.Unit` removed by
        this function will be removed from the hierarchy as well, even
        if they are still linked from a :class:`neo.core.Unit` or
        :class:`neo.core.Segment`, respectively. In this case, their
        links to the hierarchy defined by ``obj`` will be kept intact.
    """
    classname = type(obj).__name__

    # Parent for arbitrary object
    if HAS_DESCRIPTION:
        if classname in neo.description.many_to_one_relationship:
            for n in neo.description.many_to_one_relationship[classname]:
                p = getattr(obj, n.lower())
                if p is None:
                    continue
                l = getattr(p, classname.lower() + 's', ())
                try:
                    l.remove(obj)
                except ValueError:
                    pass
    else:
        for n in obj._single_parent_objects:
            p = getattr(obj, n.lower())
            if p is None:
                continue
            l = getattr(p, classname.lower() + 's', ())
            try:
                l.remove(obj)
            except ValueError:
                pass

    # Many-to-many relationships
    if isinstance(obj, neo.RecordingChannel):
        for rcg in obj.recordingchannelgroups:
            try:
                idx = rcg.recordingchannels.index(obj)
                if rcg.channel_indexes.shape[0] == len(rcg.recordingchannels):
                    rcg.channel_indexes = sp.delete(rcg.channel_indexes, idx)
                if rcg.channel_names.shape[0] == len(rcg.recordingchannels):
                    rcg.channel_names = sp.delete(rcg.channel_names, idx)
                rcg.recordingchannels.remove(obj)
            except ValueError:
                pass

    if isinstance(obj, neo.RecordingChannelGroup):
        for rc in obj.recordingchannels:
            try:
                rc.recordingchannelgroups.remove(obj)
            except ValueError:
                pass

    _handle_orphans(obj, remove_half_orphans)


def extract_spikes(train, signals, length, align_time):
    """ Extract spikes with waveforms from analog signals using a spike train. 
    Spikes that are too close to the beginning or end of the shortest signal
    to be fully extracted are ignored.

    :type train: :class:`neo.core.SpikeTrain`
    :param train: The spike times.
    :param sequence signals: A sequence of :class:`neo.core.AnalogSignal`
        objects from which the spikes are extracted. The waveforms of
        the returned spikes are extracted from these signals in the
        same order they are given.
    :type length: Quantity scalar
    :param length: The length of the waveform to extract as time scalar.
    :type align_time: Quantity scalar
    :param align_time: The alignment time of the spike times as time scalar.
        This is the time delta from the start of the extracted waveform
        to the exact time of the spike.
    :returns: A list of :class:`neo.core.Spike` objects, one for each
        time point in ``train``. All returned spikes include their
        ``waveform`` property.
    :rtype: list
    """
    if not signals:
        raise ValueError('No signals to extract spikes from')
    ref = signals[0]
    for s in signals[1:]:
        if ref.sampling_rate != s.sampling_rate:
            raise ValueError(
                'All signals for spike extraction need the same sampling rate')

    wave_unit = signals[0].units
    srate = signals[0].sampling_rate
    end = min(s.shape[0] for s in signals)

    aligned_train = train - align_time
    cut_samples = int((length * srate).simplified)

    st = sp.asarray((aligned_train * srate).simplified)

    # Find extraction epochs
    st_ok = (st >= 0) * (st < end - cut_samples)
    epochs = sp.vstack((st[st_ok], st[st_ok] + cut_samples)).T.astype(sp.int64)

    nspikes = epochs.shape[0]
    if not nspikes:
        return []

    # Create data
    data = sp.vstack([sp.asarray(s.rescale(wave_unit)) for s in signals])
    nc = len(signals)

    spikes = []
    for s in xrange(nspikes):
        waveform = sp.zeros((cut_samples, nc))
        for c in xrange(nc):
            waveform[:, c] = \
                data[c, epochs[s, 0]:epochs[s, 1]]
        spikes.append(neo.Spike(train[st_ok][s], waveform=waveform * wave_unit,
                                sampling_rate=srate))

    return spikes
