
import quantities as pq
import rate_estimation
import scipy as sp
import signal_processing as sigproc


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
    13(4), 751-763.* This implementation is normalized to yield a distance of
    1.0 for the distance between an empty spike train and a spike train with a
    single spike. Divide the result by sqrt(2.0) to get the normalization used
    in the cited paper.

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
        if markage[u].size <= 0:
            continue
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
            js, ks = _searchsorted_pairwise(trains[u], trains[v])
            start_j = sp.searchsorted(js, 0)
            start_k = sp.searchsorted(ks, 0)
            for i, j in enumerate(js[start_j:], start_j):
                D[u, v] -= (2.0 * inv_exp_trains[u][i] * exp_trains[v][j] *
                            (1.0 + markage[v][j]))
            for i, k in enumerate(ks[start_k:], start_k):
                D[u, v] -= (2.0 * inv_exp_trains[v][i] * exp_trains[u][k] *
                            (1.0 + markage[u][k]))
            D[v, u] = D[u, v]

    return sp.sqrt(D)


def _searchsorted_pairwise(a, b):
    """ Find indices for both of the two sequences where elements from one
    sequence should be inserted into the other sequence to maintain order.

    If values in `a` and `b` are equal, the values in `b` will always be
    considered as smaller.

    :param sequence a: A sorted sequence.
    :param sequence b: A sorted sequence.
    :returns: The indices for insertion of `a` into `b` and for insertion of `b`
        into `a`
    :rtype: Tuple of arrays.
    """

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


def st_inner(
        a, b, kernel, kernel_area_fraction=0.99999, sampling_rate=None):
    """ Calculates the inner product of two spike trains given a kernel.

    Let :math:`v_a(t)` and :math:`v_b(t)` with :math:`t \\in \\mathcal{T}` be
    the spike trains convolved with some kernel. Then, the inner product of the
    spike trains is defined as :math:`\\int_{\\mathcal{T}} v_a(t)v_b(t) dt`.

    Further information can be found in *Paiva, A. R. C., Park, I., & Principe,
    J. (2010). Inner products for representation and learning in the spike
    train domain. Statistical Signal Processing for Neuroscience and
    Neurotechnology, Academic Press, New York.*

    :param SpikeTrain a: First spike train.
    :param SpikeTrain b: Second spike train.
    :param kernel: Kernel to be convolved with the spike trains.
    :type kernel: :class:`.signal_processing.Kernel`
    :param float kernel_area_fraction: A value between 0 and 1 which controls
        the interval over which the kernel will be discretized. At least the
        given fraction of the complete kernel area will be covered. Higher
        values can lead to more accurate results (besides the sampling rate).
    :param float kernel_area_fraction:
    :param sampling_rate: The sampling rate which will be used to bin
        the spike trains. If `None`, the maximum sampling rate stored in the
        two spike trains will be used. If it is also `None` for both spike
        trains, that, :py:const:`signal_processing.default_sampling_rate`
        will be used.
    :type sampling_rate: Quantity scalar
    :returns: The inner product of the two spike trains given the kernel.
    :rtype: Quantity scalar
    """

    if sampling_rate is None:
        sampling_rate = max(a.sampling_rate, b.sampling_rate)
        if sampling_rate is None or sampling_rate <= 0 * pq.Hz:
            sampling_rate = sigproc.default_sampling_rate

    t_start, t_stop = rate_estimation.minimum_spike_train_interval({0: (a, b)})
    padding = kernel.boundary_enclosing_at_least(kernel_area_fraction)
    t_start -= 2 * padding
    t_stop += 2 * padding

    conv_a, _ = sigproc.st_convolve(
        a, kernel, kernel_area_fraction, t_start=t_start, t_stop=t_stop,
        mode='full', sampling_rate=sampling_rate)
    if a is b:
        conv_b = conv_a
    else:
        conv_b, _ = sigproc.st_convolve(
            b, kernel, kernel_area_fraction, t_start=t_start, t_stop=t_stop,
            mode='full', sampling_rate=sampling_rate)
    return (sp.inner(conv_a, conv_b) * conv_a.units * conv_b.units
            / sampling_rate)


def st_norm(train, kernel, **inner_params):
    return st_inner(train, train, kernel, **inner_params) ** 0.5
