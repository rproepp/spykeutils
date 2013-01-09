
import quantities as pq
import scipy as sp


def victor_purpura_dist(a, b, q=1.0 * pq.s ** -1):
    """ Calculates the Victor-Purpura's (VP) distance. """
    # TODO documentation
    if a.size <= 0 or b.size <= 0:
        return max(a.size, b.size)

    if a.size < b.size:
        a, b = b, a

    cost_a = sp.arange(float(max(1, a.size)) + 1)
    cost_b = sp.arange(float(max(1, b.size)) + 1)

    for num_spikes_processed in xrange(b.size):
        cost_a[0] = cost_b[0] = min(
            cost_b[1] + 1, cost_a[1] + 1, cost_a[0] + q * abs(
                a[num_spikes_processed] - b[num_spikes_processed]))
        for i in xrange(1, cost_a.size - num_spikes_processed - 1):
            cost_a[i] = min(
                cost_a[i - 1] + 1, cost_a[i + 1] + 1,
                cost_a[i] + q * abs(
                    a[num_spikes_processed + i] - b[num_spikes_processed]))
        for j in xrange(1, cost_b.size - num_spikes_processed - 1):
            cost_b[j] = min(
                cost_b[j - 1] + 1, cost_b[j + 1] + 1,
                cost_b[j] + q * abs(
                    a[num_spikes_processed] - b[num_spikes_processed + j]))

    return cost_a[-cost_b.size]
