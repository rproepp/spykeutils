import neo
import numpy.random
import quantities as pq
import scipy as sp
import _scipy_quantities as spq


def gen_homogeneous_poisson(
        rate, t_start=0 * pq.s, t_stop=None, max_spikes=None,
        refractory=0 * pq.s):
    """ Generate a homogeneous Poisson spike train. The length is controlled
    with `t_stop` and `max_spikes`. Either one or both of these arguments have
    to be given.

    :param rate: Average firing rate of the spike train to generate as
        frequency scalar.
    :type rate: Quantity scalar
    :param t_start: Time at which the spike train begins as time scalar. The
        first actual spike will be greater than this time.
    :type t_start: Quantity scalar
    :param t_stop: Time at which the spike train ends as time scalar. All
        generated spikes will be lower or equal than this time. If set to None,
        the number of generated spikes is controlled by `max_spikes` and
        `t_stop` will be equal to the last generated spike.
    :type t_stop: Quantity scalar
    :param max_spikes: Maximum number of spikes to generate. Fewer spikes might
        be generated in case `t_stop` is also set.
    :param refractory: Absolute refractory period as time scalar. No spike will
        follow another spike for the given duration. Afterwards the firing rate
        will instantaneously be set to `rate` again.
    :type refractory: Quantity scalar

    :returns: The generated spike train.
    :rtype: :class:`neo.core.SpikeTrain`
    """

    if t_stop is None and max_spikes is None:
        raise ValueError('Either t_stop or max_spikes has to be set.')

    if max_spikes is not None:
        spike_times = sp.cumsum(numpy.random.exponential(
            rate ** -1, max_spikes)) * (rate.units ** -1).simplified
        spike_times += t_start
        if refractory > 0:
            spike_times += sp.arange(spike_times.size) * refractory
        if t_stop is not None:
            spike_times = spike_times[spike_times <= t_stop]
    else:
        scale = (rate ** -1).rescale(t_stop.units)
        trains = []
        last_spike = t_start.rescale(t_stop.units)
        while last_spike < t_stop:
            # Generate a bit more than the average number of expected spike to
            # be finished in most cases in one loop. The factor was determined
            # empirically.
            num_spikes = int(1.7 * (
                (t_stop - last_spike) * rate).simplified) + 1
            train = sp.cumsum(numpy.random.exponential(scale, num_spikes)) * \
                scale.units + last_spike
            if refractory > 0:
                train += sp.arange(train.size) * refractory
            if train.size > 0:
                last_spike = train[-1]
                if last_spike >= t_stop:
                    train = train[train < t_stop]
                trains.append(train)
        spike_times = spq.concatenate(trains)

    if t_stop is None:
        t_stop = spike_times[-1]
    return neo.SpikeTrain(spike_times, t_start=t_start, t_stop=t_stop)


def gen_inhomogeneous_poisson(
        modulation, max_rate, t_start=0 * pq.s, t_stop=None, max_spikes=None,
        refractory=0 * pq.s):
    """ Generate an inhomogeneous Poisson spike train. The length is controlled
    with `t_stop` and `max_spikes`. Either one or both of these arguments have
    to be given.

    :param function modulation: Function :math:`f((t_1, \\dots, t_n)):
        [\\text{t\\_start}, \\text{t\\_end}]^n \\rightarrow [0, 1]^n` giving
        the instantaneous firing rates at times :math:`(t_1, \\dots, t_n)` as
        proportion of `max_rate`. Thus, a 1-D array will be passed to the
        function and it should return an array of the same size.
    :param max_rate: Maximum firing rate of the spike train to generate as
        frequency scalar.
    :type max_rate: Quantity scalar
    :param t_start: Time at which the spike train begins as time scalar. The
        first actual spike will be greater than this time.
    :type t_start: Quantity scalar
    :param t_stop: Time at which the spike train ends as time scalar. All
        generated spikes will be lower or equal than this time. If set to None,
        the number of generated spikes is controlled by `max_spikes` and
        `t_stop` will be equal to the last generated spike.
    :type t_stop: Quantity scalar
    :param refractory: Absolute refractory period as time scalar. No spike will
        follow another spike for the given duration. Afterwards the firing rate
        will instantaneously be set to `rate` again.
    :type refractory: Quantity scalar

    :returns: The generated spike train.
    :rtype: :class:`neo.core.SpikeTrain`
    """

    st = gen_homogeneous_poisson(
        max_rate, t_start, t_stop, max_spikes, refractory)
    return st[numpy.random.rand(st.size) < modulation(st)]
