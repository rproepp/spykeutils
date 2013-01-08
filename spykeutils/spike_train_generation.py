
import neo
import numpy.random
import quantities as pq
import scipy as sp


def gen_homogeneous_poisson(
        rate, t_start=0 * pq.s, t_stop=None, max_spikes=None):
    """ Generate a homogeneous Poisson spike train. The length is controlled
    with `t_stop` and `max_spikes`. Either one or both of these arguments have
    to be given.

    :param rate: Average firing rate of the spike train to generate.
    :type rate: Quantity scalar
    :param t_start: Time at which the spike train begins. The first actual spike
        will be greater than this time.
    :type t_start: Quantity scalar
    :param t_stop: Time at which the spike train ends. All generated spikes will
        be lower or equal than this time. If set to None, the number of
        generated spikes is controlled by `max_spikes` and `t_stop` will be
        equal to the last generated spike.
    :type t_stop: Quantity scalar
    :param max_spikes: Maximum number of spikes to generate. Fewer spikes might
        be generated in case `t_stop` is also set.

    :returns: The generated spike train.
    :rtype: SpikeTrain
    """

    if t_stop is None and max_spikes is None:
        raise ValueError('Either t_stop or max_spikes has to be set.')

    if max_spikes is not None:
        spike_times = sp.cumsum(numpy.random.exponential(
            rate ** -1, max_spikes)) * (rate.units ** -1).simplified
        spike_times += t_start
        if t_stop is not None:
            spike_times = spike_times[spike_times <= t_stop]
    else:
        scale = (rate ** -1).rescale(t_stop.units)
        spike_times = [t_start]
        while spike_times[-1] <= t_stop:
            spike_times.append(
                spike_times[-1] + numpy.random.exponential(scale) * scale.units)
        spike_times = spike_times[1:-1] * scale.units

    if t_stop is None:
        t_stop = spike_times[-1]
    return neo.SpikeTrain(spike_times, t_start=t_start, t_stop=t_stop)
