import neo

def spikes_from_spike_train(spike_train):
    """ Return a list of spikes for a spike train.
    :param SpikeTrain spike_train: A spike train with spike waveform
        data. If no such data is found, an empty list is returned.
    :returns: A list of Spike objects, one for every spike in ``spike_train``.
    :rtype: list
    """
    waves = spike_train.waveforms
    if waves is None:
        return []

    spikes = []
    for i, t in enumerate(spike_train):
        s = neo.Spike(t, sampling_rate=spike_train.sampling_rate,
            waveform=waves[i, :, :], left_sweep=spike_train.left_sweep)
        s.unit = spike_train.unit
        s.segment = spike_train.segment
        spikes.append(s)

    return spikes