
import neo
import quantities as pq
import scipy as sp


def create_empty_spike_train(t_start=0.0 * pq.s, t_stop=10.0 * pq.s):
    return neo.SpikeTrain(sp.array([]) * pq.s, t_start=t_start, t_stop=t_stop)

def arange_spikes(t_start, t_stop=None, t_step=1.0 * pq.s):
    """ Arranges equally spaced spikes in a spike train over an interval. There
    will be no spikes at the interval boundaries.

    :param t_start: The time point of the interval if `t_stop` is not `None`.
        Otherwise, it will be end point of the interval and the start point will
        be set to 0s.
    :type t_start: Quantity scalar
    :param t_stop: The end point of the interval.
    :type t_stop: Quantity scalar
    :param t_step: Spacing between the spikes.
    :type t_step: Quantity scalar
    :returns The arranged spike train.
    :rtype: :class:`neo.SpikeTrain`
    """

    if t_stop is None:
        t_stop = t_start
        t_start = 0.0 * pq.s

    t_start.units = t_step.units
    t_stop.units = t_step.units

    spike_times = sp.arange(t_start + t_step, t_stop, t_step) * t_step.units
    return neo.SpikeTrain(spike_times, t_start=t_start, t_stop=t_stop)
