
import quantities as pq
import scipy as sp


def victor_purpura_dist(a, b, q=1.0 * pq.s ** -1):
    """ Calculates the Victor-Purpura's (VP) distance. It is often denoted as
    :math:`D^{\\text{spike}}[q]`.

    It is defined as the minimal cost of transforming spike train `a` into
    spike train `b` by using the following operations:

        * Inserting or deleting a spike (cost 1.0).
        * Shifting a spike from :math:`t` to :math:`t'` (cost :math:`q \\cdot |t
          - t'|`).

    A detailed description can be found in
    *Victor, J. D., & Purpura, K. P. (1996). Nature and precision of temporal
    coding in visual cortex: a metric-space analysis. Journal of
    Neurophysiology.*

    Given the number of spikes :math:`n_a` and :math:`n_b` in the spike trains
    the run-time complexity of this function is :math:`O(n_a n_b)`
    and :math:`O(n_a + n_b)` memory will be needed.

    :param SpikeTrain a:
    :param SpikeTrain b:
    :param q: Cost factor for spike shifts.
    :type q: Quantity scalar
    :rtype: float
    """
    if a.size <= 0 or b.size <= 0:
        return max(a.size, b.size)

    if a.size < b.size:
        a, b = b, a

    # The algorithm used is based on the one given in
    #
    # Victor, J. D., & Purpura, K. P. (1996). Nature and precision of temporal
    # coding in visual cortex: a metric-space analysis. Journal of
    # Neurophysiology.
    #
    # It constructs a matrix cost[i, j] containing the minimal cost when only
    # considering the first i and j spikes of the spike trains (the reference
    # given above denotes this matrix with G). However, one never needs to
    # store more than one row and one column at the same time for calculating
    # the VP distance. cost_a[0:cost_a.size - num_spikes_processed] corresponds
    # to cost[num_spikes_processed:, num_spikes_processed]. The same holds for
    # cost_b by replacing the occurences of cost_a.

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
