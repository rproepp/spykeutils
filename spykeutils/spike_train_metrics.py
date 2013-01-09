
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


def van_rossum_dist(trains, tau=1.0 * pq.s):
    """ Calculates the van Rossum distance.

    It is defined as Euclidean distance of the spike trains convolved with a
    causal decaying exponential function. A detailed description can be found in
    *Rossum, M. C. W. (2001). A novel spike distance. Neural Computation,
    13(4), 751-763.*

    Given :math:`N` spike trains with :math:`n` spikes on average the run-time
    complexity of this function is :math:`O(N^2 n)` and :math:`O(N^2 + Nn^2)`
    memory will be needed.

    :param sequence trains: SpikeTrains of which the van Rossum distance will be
        calculated pairwise.
    :param tau: Decay rate of the exponential function. Controls for which time
        scale the metric will be sensitive.
    :type tau: Quantity scalar
    :returns: Matrix containing the van Rossum distances for all pairs of spike
        trains.
    :rtype: 2-D array
    """

    # The implementation is based on
    #
    # Houghton, C., & Kreuz, T. (2012). On the efficient calculation of van
    # Rossum distances. Network: Computation in Neural Systems, 23(1-2), 48-58.
    #
    # Note that the cited paper contains some errors.

    exp_trains = [sp.exp(st / tau) for st in trains]
    inv_exp_trains = [1.0 / st for st in exp_trains]
    exp_diffs = [sp.outer(exp_train, inv_exp_train) for
                 exp_train, inv_exp_train in zip(exp_trains, inv_exp_trains)]

    markage = [sp.empty(st.size) for st in trains]
    for u in xrange(len(markage)):
        markage[u][0] = 0
        for i in xrange(1, markage[u].size):
            markage[u][i] = (markage[u][i - 1] + 1.0) * exp_diffs[u][i - 1, i]

    # Same spike train terms
    D = sp.zeros((len(trains), len(trains)))
    for u in xrange(D.shape[0]):
        summand = markage[u].size + 2.0 * sp.sum(markage[u])
        D[u, :] += summand
        D[:, u] += summand

    # Cross spike train terms
    for u in xrange(D.shape[0]):
        for v in xrange(u, D.shape[1]):
            js, ks = searchsorted_pairwise(trains[u], trains[v])
            start_j = sp.searchsorted(js, 0)
            start_k = sp.searchsorted(ks, 0)
            for i, j in enumerate(js[start_j:], start_j):
                D[u, v] -= (2.0 * inv_exp_trains[u][i] * exp_trains[v][j] *
                            (1.0 + markage[v][j]))
            for i, k in enumerate(ks[start_k:], start_k):
                D[u, v] -= (2.0 * inv_exp_trains[v][i] * exp_trains[u][k] *
                            (1.0 + markage[u][k]))
            D[v, u] = D[u, v]

    return D


def searchsorted_pairwise(a, b):
    idx_a = sp.empty(len(a))
    idx_b = sp.empty(len(b))
    i = j = 0
    while i < len(a) and j < len(b):
        if a[i] < b[j]:
            idx_a[i] = j - 1
            i += 1
        elif a[i] >= b[j]:
            idx_b[j] = i - 1
            j += 1
    idx_a[i:] = j - 1
    idx_b[j:] = i - 1
    return idx_a, idx_b
