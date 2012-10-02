import scipy as sp
import neo

from spyke_exception import SpykeException

def spikes_from_spike_train(spike_train):
    """ Return a list of spikes for a spike train.

    Note that while the created spikes have references to the same segment and
    unit as the spike train, the relationships in the other direction are
    not automatically created (the spikes are not attached to the unit or
    segment).

    :param SpikeTrain spike_train: A spike train with spike waveform
        data. If no such data is found, an empty list is returned.
    :returns: A list of Spike objects, one for every spike in ``spike_train``.
    :rtype: list
    """
    waves = spike_train.waveforms

    spikes = []
    for i, t in enumerate(spike_train):
        s = neo.Spike(t, sampling_rate=spike_train.sampling_rate,
            left_sweep=spike_train.left_sweep)
        if waves is not None:
            s.waveform=waves[i, :, :]
        s.unit = spike_train.unit
        s.segment = spike_train.segment
        spikes.append(s)

    return spikes

def spike_train_from_spikes(spikes, include_waveforms=True):
    """ Return a spike train for a list of spikes.

    All spikes must have an identical left sweep, the same unit and the same
    segment, otherwise a ``SpykeException`` is raised.

    Note that while the created spike train has references to the same
    segment and unit as the spikes, the relationships in the other direction
    are not automatically created (the spike train is not attached to the
    unit or segment).

    :param spikes: A list of spike objects.
    :param bool include_waveforms: Determines if the waveforms from the Spike
        objects are used to fill the ``waveforms`` property of the resulting
        spike train. If ``True``, all spikes need a ``waveform`` property
        with the same shape or a ``SpykeException`` is raised (or the
        ``waveform`` property needs to be ``None`` for all spikes).
    :return: A SpikeTrain object including all elements of ``spikes``.
    :rtype: :class:`neo.core.SpikeTrain`
    """
    if not spikes:
        raise SpykeException('No spikes to create spike train!')

    tu = spikes[0].time.units
    times = sp.zeros(len(spikes)) * tu
    s = spikes[0].segment
    u = spikes[0].unit
    ls = spikes[0].left_sweep

    if include_waveforms and spikes[0].waveform is not None:
        sh = spikes[0].waveform.shape
        wu = spikes[0].waveform.units
        waves = sp.zeros((len(spikes), sh[0], sh[1])) * wu
    else:
        waves = None
        sh = None

    for i, spike in enumerate(spikes):
        if (u != spike.unit or s != spike.segment or
            ls != spike.left_sweep):
            raise SpykeException('Cannot create spike train from spikes with' +
                'nonuniform properties!')

        times[i] = spikes[i].time

        if include_waveforms:
            if spike.waveform is None:
                if waves is not None:
                    raise SpykeException('Cannot create spike train from ' +
                                         'spikes where some waveforms are None')
            elif sh != spike.waveform.shape:
                raise SpykeException('Cannot create spike train from spikes ' +
                                     'with nonuniform waveform shapes!')
            if waves is not None:
                waves[i,:,:] = spike.waveform

    return neo.SpikeTrain(times, times.max(), waveforms=waves, left_sweep=ls)


def analog_signals_from_analog_signal_array(signal_array):
    """ Return a list of analog signals an analog signal array.

    If ``signal_array`` is attached to a recording channel group with exactly
    is many channels as there are channels in ``signal_array``, each created
    signal will be assigned the corresponding channel. If the attached
    recording channel group has only one recording channel, all created signals
    will be assigned to this channel. In all other cases, the created
    signal will not have a reference to a recording channel.

    Note that while the created signals may have references to a segment and
    channels, the relationships in the other direction are
    not automatically created (the signals are not attached to the recording
    channel or segment).

    :param signal_array: An analog signal array.
    :type signal_array: :class:`neo.core.AnalogSignalArray`
    :return: A list of analog signals, one for every channel in
        ``signal_array``
    :rtype: list
    """
    signals = []
    rcg = signal_array.recordingchannelgroup

    for i in xrange(signal_array.shape[1]):
        s = neo.AnalogSignal(signal_array[:,i], t_start = signal_array.t_start,
            sampling_rate=signal_array.sampling_rate)
        if len(rcg.recordingchannels) == 1:
            s.recordingchannel = rcg.recodingchannels[0]
        elif len(rcg.recordingchannels) == signal_array.shape[1]:
            s.recordingchannel = rcg.recodingchannels[i]
        s.segment = signal_array.segment
        signals.append(s)

    return signals

