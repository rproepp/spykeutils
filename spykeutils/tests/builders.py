
import neo
import quantities as pq
import scipy as sp


def create_empty_spike_train(t_start=0.0 * pq.s, t_stop=10.0 * pq.s):
    return neo.SpikeTrain(sp.array([]) * pq.s, t_start=t_start, t_stop=t_stop)
