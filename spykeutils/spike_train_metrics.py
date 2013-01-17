
import heapq
import quantities as pq
import rate_estimation
import scipy as sp
import signal_processing as sigproc


def _dicts_to_lists(dicts, keys_to_extract):
    return ([d[k] for k in keys_to_extract] for d in dicts)


def _merge_trains_and_label_spikes(trains):
    labeled_trains = (zip(st, len(st) * (label,)) for label, st
                      in enumerate(trains))
    return list(heapq.merge(*labeled_trains))


def victor_purpura_multiunit_dist(
        a, b, reassignment_cost, q=1.0 * pq.Hz, kernel=None):
    """ Calculates the Victor-Purpura's (VP) multi-unit distance.

    It is defined as the minimal cost of transforming the spike trains `a` into
    spike trains `b` by using the following operations:

        * Inserting or deleting a spike (cost 1.0).
        * Shifting a spike from :math:`t` to :math:`t'` (cost :math:`q \\cdot |t
          - t'|`).
        * Moving a spike to another spike train (cost `reassignment_cost`).

    A detailed description can be found in
    *Aronov, D. (2003). Fast algorithm for the metric-space analysis of
    simultaneous responses of multiple single neurons. Journal of Neuroscience
    Methods.*

    Given the average number of spikes :math:`N` in a spike train and :math:`L`
    spike trains the run-time and memory complexity are :math:`O(LN^{L+1})`

    For calculating the distance between only two spike trains one should use
    :func:`victor_purpura_dist` which is more memory efficient.

    :param dict a: Dictionary of spike trains.
    :param dict b: Dictionary of spike trains. Must have the same set of keys as
        `a`.
    :param float reassignment_cost: Cost to reassign a spike from one train to
        another (sometimes denoted with :math:`k`). Should be between 0 and 2.
        For 0 spikes can be reassigned without any cost, for 2 and above it is
        cheaper to delete and reinsert a spike.
    :param q: Cost factor for spike shifts. If `kernel` is not `None`, `q` will
        be ignored.
    :type q: Quantity scalar
    :param kernel: Kernel to use in the calculation of the distance. If
        `kernel` is `None`, an unnormalized triangular kernel with a half width
        of `2.0/q` will be used.
    :type kernel: :class:`.signal_processing.Kernel`
    :rtype: float
    """

    if len(a) != len(b):
        raise ValueError("Number of spike trains in a and b differs.")
    a, b = _dicts_to_lists((a, b), a.keys())

    if kernel is None:
        kernel = sigproc.TriangularKernel(2.0 / q, normalize=False)

    # The algorithm used is based on the one given in
    #
    # Victor, J. D., & Purpura, K. P. (1996). Nature and precision of temporal
    # coding in visual cortex: a metric-space analysis. Journal of
    # Neurophysiology.
    #
    # It constructs a matrix cost[i, j_1, ... j_L] containing the minimal cost
    # when only considering the first i spikes of the merged spikes of a and
    # j_w spikes of the spike trains of b (the reference given above denotes
    # this matrix with G).

    # The algorithm is not asymmetric, swap a and b if that will save us time.
    a_num_spikes = [st.size for st in a]
    b_num_spikes = [st.size for st in b]
    complexity_same = sp.sum(a_num_spikes) * sp.prod(b_num_spikes)
    complexity_swapped = sp.prod(a_num_spikes) * sp.sum(b_num_spikes)
    if complexity_swapped < complexity_same:
        a, b = b, a
        a_num_spikes, b_num_spikes = b_num_spikes, a_num_spikes

    a_merged = _merge_trains_and_label_spikes(a)
    b_dims = sp.asarray(b_num_spikes) + 1
    cost = sp.empty((sp.sum(a_num_spikes) + 1,) + tuple(b_dims))
    cost[(sp.s_[:],) + len(b) * (0,)] = sp.arange(cost.shape[0])
    cost[0, ...] = sp.sum(sp.indices(b_dims), axis=0)

    for a_idx in xrange(1, cost.shape[0]):
        a_spike_time = a_merged[a_idx - 1][0]
        a_spike_label = a_merged[a_idx - 1][1]

        b_idx_iter = sp.ndindex(*b_dims)
        b_idx_iter.next()  # cost[:, 0, ..., 0] has already been initialized
        for b_idx in b_idx_iter:
            # Generate set of indices
            # {(j_1, ..., j_w - 1, ... j_L) | 1 <= w <= L}
            # and determine the calculated cost for each element.
            b_origin_indices = [
                tuple(sp.atleast_1d(sp.squeeze(idx))) for idx in sp.split(
                    sp.asarray(b_idx) - sp.eye(len(b_idx)), len(b_idx), axis=1)]
            invalid_origin_indices = sp.asarray(b_idx) == 0
            origin_costs = cost[[a_idx - 1] + b_origin_indices]
            origin_costs[invalid_origin_indices] = sp.inf

            b_spike_label = sp.argmin(origin_costs)
            b_spike_time = b[b_spike_label][b_idx[b_spike_label] - 1]
            cost_shift = origin_costs[b_spike_label] + \
                2 - 2 * kernel(a_spike_time - b_spike_time).simplified + \
                reassignment_cost * (a_spike_label != b_spike_label)

            cost_delete_in_a = cost[(a_idx - 1,) + b_idx] + 1
            if sp.all(invalid_origin_indices):
                cost_delete_in_b = sp.inf
            else:
                cost_delete_in_b = sp.amin(
                    cost[[a_idx] + b_origin_indices]
                    [sp.logical_not(invalid_origin_indices) != 0]) + 1

            cost[(a_idx,) + b_idx] = min(
                cost_delete_in_b, cost_delete_in_a, cost_shift)

    return cost.flat[-1]


def victor_purpura_dist(a, b, q=1.0 * pq.Hz, kernel=None):
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
    :param q: Cost factor for spike shifts. If `kernel` is not `None`, `q` will
        be ignored.
    :type q: Quantity scalar
    :param kernel: Kernel to use in the calculation of the distance. If
        `kernel` is `None`, an unnormalized triangular kernel with a half width
        of `2.0/q` will be used.
    :type kernel: :class:`.signal_processing.Kernel`
    :rtype: float
    """

    if a.size <= 0 or b.size <= 0:
        return max(a.size, b.size)

    if a.size < b.size:
        a, b = b, a

    if kernel is None:
        kernel = sigproc.TriangularKernel(2.0 / q, normalize=False)

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
    # cost_b by replacing the occurrences of cost_a.

    cost_a = sp.arange(float(max(1, a.size)) + 1)
    cost_b = sp.arange(float(max(1, b.size)) + 1)

    for num_spikes_processed in xrange(b.size):
        cost_a[0] = cost_b[0] = min(
            cost_b[1] + 1, cost_a[1] + 1, cost_a[0] + 2 - 2 * kernel(
                a[num_spikes_processed] - b[num_spikes_processed]).simplified)
        for i in xrange(1, cost_a.size - num_spikes_processed - 1):
            cost_a[i] = min(
                cost_a[i - 1] + 1, cost_a[i + 1] + 1,
                cost_a[i] + 2 - 2 * kernel(
                    a[num_spikes_processed + i] -
                    b[num_spikes_processed]).simplified)
        for j in xrange(1, cost_b.size - num_spikes_processed - 1):
            cost_b[j] = min(
                cost_b[j - 1] + 1, cost_b[j + 1] + 1,
                cost_b[j] + 2 - kernel(
                    a[num_spikes_processed] -
                    b[num_spikes_processed + j]).simplified)

    return cost_a[-cost_b.size]


def van_rossum_dist(trains, tau=1.0 * pq.s, kernel=None):
    """ Calculates the van Rossum distance.

    It is defined as Euclidean distance of the spike trains convolved with a
    causal decaying exponential smoothing filter. A detailed description can be
    found in *Rossum, M. C. W. (2001). A novel spike distance. Neural
    Computation, 13(4), 751-763.* This implementation is normalized to yield
    a distance of 1.0 for the distance between an empty spike train and a spike
    train with a single spike. Divide the result by sqrt(2.0) to get the
    normalization used in the cited paper.

    Given :math:`N` spike trains with :math:`n` spikes on average the run-time
    complexity of this function is :math:`O(N^2 n)` and :math:`O(N^2 + Nn^2)`
    memory will be needed, if the default Laplacian kernel is used (which
    corresponds to the causal decaying exponential smoothing function). Other
    kernels have probably a worse performance.

    :param sequence trains: SpikeTrains of which the van Rossum distance will be
        calculated pairwise.
    :param tau: Decay rate of the exponential function. Controls for which time
        scale the metric will be sensitive. This parameter will be ignored if
        `kernel` is not `None`. May also be `inf` which will lead to only
        measuring differences in spike count.
    :type tau: Quantity scalar
    :param kernel: Kernel to use in the calculation of the distance. This is not
        the smoothing filter, but its autocorrelation. If `kernel` is `None`, an
        unnormalized Laplacian kernel with a size of `tau` will be used.
    :type kernel: :class:`.signal_processing.Kernel`
    :returns: Matrix containing the van Rossum distances for all pairs of spike
        trains.
    :rtype: 2-D array
    """

    if kernel is None:
        if tau == sp.inf:
            spike_counts = [st.size for st in trains]
            return sp.absolute(spike_counts - sp.atleast_2d(spike_counts).T)
        kernel = sigproc.LaplacianKernel(tau, normalize=False)

    k_dist = kernel.summed_dist_matrix(trains)
    vr_dist = sp.empty_like(k_dist)
    for i, j in sp.ndindex(*k_dist.shape):
        vr_dist[i, j] = (
            k_dist[i, i] + k_dist[j, j] - k_dist[i, j] - k_dist[j, i])
    return sp.sqrt(vr_dist)


def van_rossum_multiunit_dist(a, b, weighting, tau=1.0 * pq.s, kernel=None):
    """ Calculates the van Rossum multi-unit distance.

    The single-unit distance is defined as Euclidean distance of the spike
    trains convolved with a causal decaying exponential smoothing filter.
    A detailed description can be found in *Rossum, M. C. W. (2001). A novel
    spike distance. Neural Computation, 13(4), 751-763.* This implementation is
    normalized to yield a distance of 1.0 for the distance between an empty
    spike train and a spike train with a single spike. Divide the result by
    sqrt(2.0) to get the normalization used in the cited paper.

    Given the :math:`p`- and :math:`q`-th spike train of `a` and respectively
    `b` let :math:`R_{pq}` be the squared single-unit distance between these
    two spike trains. Then the multi-unit distance is :math:`\\sqrt{\\sum_p
    (R_{pp} + c \\cdot \\sum_{q \\neq p} R_{pq})}` with :math:`c` being
    equal to `weighting`. The weighting parameter controls the interpolation
    between a labeled line and a summed population coding.

    More information can be found in
    *Houghton, C., & Kreuz, T. (2012). On the efficient calculation of van
    Rossum distances. Network: Computation in Neural Systems, 23(1-2), 48-58.*

    Given :math:`N` spike trains in total with :math:`n` spikes on average the
    run-time complexity of this function is :math:`O(N^2 n)` and :math:`O(N^2
    + Nn^2)` memory will be needed, if the default Laplacian kernel is used
    (which corresponds to the causal decaying exponential smoothing function).
    Other kernels have probably a worse performance.

    :param dict a: Dictionary of spike trains.
    :param dict b: Dictionary of spike trains. Must have the same set of keys as
        `a`.
    :param float weighting: Controls the interpolation between a labeled line
        and a summed population coding.
    :param tau: Decay rate of the exponential function. Controls for which time
        scale the metric will be sensitive. This parameter will be ignored if
        `kernel` is not `None`. May also be `inf` which will lead to only
        measuring differences in spike count.
    :type tau: Quantity scalar
    :param kernel: Kernel to use in the calculation of the distance. This is not
        the smoothing filter, but its autocorrelation. If `kernel` is `None`, an
        unnormalized Laplacian kernel with a size of `tau` will be used.
    :type kernel: :class:`.signal_processing.Kernel`
    :returns: Matrix containing the van Rossum distances for all pairs of spike
        trains.
    :rtype: 2-D array
    """

    if len(a) != len(b):
        raise ValueError("Number of spike trains in a and b differs.")
    a, b = _dicts_to_lists((a, b), a.keys())

    if kernel is None and tau != sp.inf:
        kernel = sigproc.LaplacianKernel(tau, normalize=False)

    if kernel is None:
        spike_counts = sp.atleast_2d([st.size for st in a + b])
        k_dist = spike_counts.T * (spike_counts - spike_counts.T)
    else:
        k_dist = kernel.summed_dist_matrix(a + b)

    non_diagonal = sp.logical_not(sp.eye(len(a)))
    summed_population = (
        sp.trace(k_dist) - sp.trace(k_dist, len(a)) - sp.trace(k_dist, -len(a)))
    labeled_line = (
        sp.sum(k_dist[:len(a), :len(a)][non_diagonal]) +
        sp.sum(k_dist[len(a):, len(a):][non_diagonal]) -
        sp.sum(k_dist[:len(a), len(a):][non_diagonal]) -
        sp.sum(k_dist[len(a):, :len(a)][non_diagonal]))
    return sp.sqrt(summed_population + weighting * labeled_line)


def st_inner(
        a, b, smoothing_filter,
        filter_area_fraction=sigproc.default_kernel_area_fraction,
        sampling_rate=None):
    """ Calculates the inner product of two spike trains given a smoothing
    filter.

    Let :math:`v_a(t)` and :math:`v_b(t)` with :math:`t \\in \\mathcal{T}` be
    the spike trains convolved with some smoothing filter. Then, the inner
    product of the spike trains is defined as :math:`\\int_{\\mathcal{T}}
    v_a(t)v_b(t) dt`.

    Further information can be found in *Paiva, A. R. C., Park, I., & Principe,
    J. (2010). Inner products for representation and learning in the spike
    train domain. Statistical Signal Processing for Neuroscience and
    Neurotechnology, Academic Press, New York.*

    :param SpikeTrain a: First spike train.
    :param SpikeTrain b: Second spike train.
    :param smoothing_filter: A smoothing filter to be convolved with the spike
        trains.
    :type smoothing_filter: :class:`.signal_processing.Kernel`
    :param float filter_area_fraction: A value between 0 and 1 which controls
        the interval over which the `smoothing_filter` will be discretized. At
        least the given fraction of the complete `smoothing_filter` area will be
        covered. Higher values can lead to more accurate results (besides the
        sampling rate).
    :param sampling_rate: The sampling rate which will be used to bin
        the spike trains. If `None`, the maximum sampling rate stored in the
        two spike trains will be used. If it is also `None` for both spike
        trains, that, :py:const:`signal_processing.default_sampling_rate`
        will be used.
    :type sampling_rate: Quantity scalar
    :returns: The inner product of the two spike trains given the smoothing_filter.
    :rtype: Quantity scalar
    """

    convolved, sampling_rate = _prepare_for_inner_prod(
        [a, b], smoothing_filter, filter_area_fraction, sampling_rate)
    return (sp.inner(*convolved) * convolved[0].units * convolved[1].units
            / sampling_rate)


def _prepare_for_inner_prod(
        trains, smoothing_filter, filter_area_fraction, sampling_rate):
    if sampling_rate is None:
        sampling_rate = max([st.sampling_rate for st in trains])
        if sampling_rate is None or sampling_rate <= 0 * pq.Hz:
            sampling_rate = sigproc.default_sampling_rate

    t_start, t_stop = rate_estimation.minimum_spike_train_interval({0: trains})
    padding = smoothing_filter.boundary_enclosing_at_least(filter_area_fraction)
    t_start -= 2 * padding
    t_stop += 2 * padding

    return [sigproc.st_convolve(
        st, smoothing_filter, filter_area_fraction, t_start=t_start,
        t_stop=t_stop, mode='full', sampling_rate=sampling_rate)[0]
        for st in trains], sampling_rate


def st_norm(
        train, smoothing_filter,
        filter_area_fraction=sigproc.default_kernel_area_fraction,
        sampling_rate=None):
    """ Calculates the spike train norm given a smoothing filter.

    Let :math:`v(t)` with :math:`t \\in \\mathcal{T}` be a spike train
    convolved with some smoothing filter. Then, the norm of the spike train is
    defined as :math:`\\int_{\\mathcal{T}} v(t)^2 dt`.

    Further information can be found in *Paiva, A. R. C., Park, I., & Principe,
    J. (2010). Inner products for representation and learning in the spike
    train domain. Statistical Signal Processing for Neuroscience and
    Neurotechnology, Academic Press, New York.*

    :param SpikeTrain train: A spike train.
    :param smoothing_filter: Smoothing filter to be convolved with the spike
        train.
    :type smoothing_filter: :class:`.signal_processing.Kernel`
    :param float filter_area_fraction: A value between 0 and 1 which controls
        the interval over which the smoothing filter will be discretized. At
        least the given fraction of the complete smoothing filter area will be
        covered. Higher values can lead to more accurate results (besides the
        sampling rate).
    :param sampling_rate: The sampling rate which will be used to bin
        the spike trains. If `None`, the maximum sampling rate stored in the
        two spike trains will be used. If it is also `None` for both spike
        trains, that, :py:const:`signal_processing.default_sampling_rate`
        will be used.
    :type sampling_rate: Quantity scalar
    :returns: The of norm the spike train given the smoothing_filter.
    :rtype: Quantity scalar
    """

    convolved, sampling_rate = _prepare_for_inner_prod(
        [train], smoothing_filter, filter_area_fraction, sampling_rate)
    return ((sp.inner(convolved[0], convolved[0]) / sampling_rate) ** 0.5 *
            convolved[0].units)


def norm_dist(
        a, b, smoothing_filter,
        filter_area_fraction=sigproc.default_kernel_area_fraction,
        sampling_rate=sigproc.default_sampling_rate):
    """ Calculates the norm distance between two spike trains given a smoothing
    filter.

    Let :math:`v_a(t)` and :math:`v_b(t)` with :math:`t \\in \\mathcal{T}` be
    the spike trains convolved with some smoothing filter. Then, the norm
    distance of the spike trains is defined as :math:`d_{ND}(a, b)
    = \\sqrt{\\int_{\\mathcal{T}} (v_a(t) - v_b(t))^2 dt}`.

    Further information can be found in *Paiva, A. R. C., Park, I., & Principe,
    J. (2010). Inner products for representation and learning in the spike
    train domain. Statistical Signal Processing for Neuroscience and
    Neurotechnology, Academic Press, New York.*

    :param SpikeTrain a: First spike train.
    :param SpikeTrain b: Second spike train.
    :param smoothing_filter: Smoothing filter to be convolved with the spike
        trains.
    :type smoothing_filter: :class:`.signal_processing.Kernel`
    :param float filter_area_fraction: A value between 0 and 1 which controls
        the interval over which the smoothing filter will be discretized. At
        least the given fraction of the complete smoothing filter area will be
        covered. Higher values can lead to more accurate results (besides the
        sampling rate).
    :param sampling_rate: The sampling rate which will be used to bin
        the spike trains. If `None`, the maximum sampling rate stored in the
        two spike trains will be used. If it is also `None` for both spike
        trains, that, :py:const:`signal_processing.default_sampling_rate`
        will be used.
    :type sampling_rate: Quantity scalar
    :returns: The norm distance of the spike trains given the smoothing_filter.
    :rtype: Quantity scalar
    """

    convolved, sampling_rate = _prepare_for_inner_prod(
        [a, b], smoothing_filter, filter_area_fraction, sampling_rate)
    return max(
        0.0 * pq.Hz,
        (sp.inner(convolved[0], convolved[0]) +
            sp.inner(convolved[1], convolved[1]) -
            2 * sp.inner(*convolved)) *
        convolved[0].units * convolved[1].units / sampling_rate) ** 0.5


def cs_dist(
        a, b, smoothing_filter,
        filter_area_fraction=sigproc.default_kernel_area_fraction,
        sampling_rate=sigproc.default_sampling_rate):
    """ Calculates the Cauchy-Schwarz distance between two spike trains given
    a smoothing filter.

    Let :math:`v_a(t)` and :math:`v_b(t)` with :math:`t \\in \\mathcal{T}` be
    the spike trains convolved with some smoothing filter and :math:`V(a, b)
    = \\int_{\\mathcal{T}} v_a(t) v_b(t) dt`. Then, the Cauchy-Schwarz distance
    of the spike trains is defined as :math:`d_{CS}(a, b) = \\arccos \\frac{V(a,
    b)^2}{V(a, a) V(b, b)}`.

    The Cauchy-Schwarz distance is closely related to the Schreiber et al.
    similarity measure :math:`S_S` by :math:`\\arccos S_S^2`

    This function numerically convolves the spike trains with the smoothing
    filter which can be quite slow and inaccurate. If the analytical result of
    the autocorrelation of the smoothing filter is known, one can use
    :func:`schreiber_similarity` for a more efficient and precise calculation.

    Further information can be found in *Paiva, A. R. C., Park, I., & Principe,
    J. (2010). Inner products for representation and learning in the spike
    train domain. Statistical Signal Processing for Neuroscience and
    Neurotechnology, Academic Press, New York.*

    :param SpikeTrain a: First spike train.
    :param SpikeTrain b: Second spike train.
    :param smoothing_filter: Smoothing filter to be convolved with the spike
        trains.
    :type smoothing_filter: :class:`.signal_processing.Kernel`
    :param float filter_area_fraction: A value between 0 and 1 which controls
        the interval over which the smoothing filter will be discretized. At
        least the given fraction of the complete smoothing filter area will be
        covered. Higher values can lead to more accurate results (besides the
        sampling rate).
    :param sampling_rate: The sampling rate which will be used to bin
        the spike trains. If `None`, the maximum sampling rate stored in the
        two spike trains will be used. If it is also `None` for both spike
        trains, that, :py:const:`signal_processing.default_sampling_rate`
        will be used.
    :type sampling_rate: Quantity scalar
    :returns: The Cauchy-Schwarz distance of the spike trains given the
        smoothing filter.
    :rtype: float
    """
    if a.size <= 0 or b.size <= 0:
        return sp.nan

    convolved, sampling_rate = _prepare_for_inner_prod(
        [a, b], smoothing_filter, filter_area_fraction, sampling_rate)
    return sp.arccos(
        sp.inner(*convolved) ** 2 / sp.inner(convolved[0], convolved[0]) /
        sp.inner(convolved[1], convolved[1]))


def schreiber_similarity(trains, kernel):
    """ Calculates the Schreiber et al. similarity measure between two spike
    trains given a kernel.

    Let :math:`v_a(t)` and :math:`v_b(t)` with :math:`t \\in \\mathcal{T}` be
    the spike trains convolved with some smoothing filter and :math:`V(a, b)
    = \\int_{\\mathcal{T}} v_a(t) v_b(t) dt`. The autocorrelation of the
    smoothing filter corresponds to the kernel used to analytically calculate
    the Schreiber et al. similarity measure. It is defined as :math:`S_{S}(a,
    b) = \\frac{V(a, b)}{\\sqrt{V(a, a) V(b, b)}}`. It is closely related to
    the Cauchy-Schwarz distance :math:`d_{CS}` by :math:`S_S = \\sqrt{\\cos
    d_{CS}}`.

    In opposite to :func:`cs_dist` which numerically convolves the spike trains
    with a smoothing filter, this function directly uses the kernel resulting
    from the smoothing filter's autocorrelation. This allows a more accurate
    and faster calculation.

    Further information can be found in:

    - *Dauwels, J., Vialatte, F., Weber, T., & Cichocki, A. (2009). On
      similarity measures for spike trains. Advances in Neuro-Information
      Processing, 177-185.*
    - *Paiva, A. R. C., Park, I., & Principe, J. C. (2009). A comparison of
      binless spike train measures. Neural Computing and Applications, 19(3),
      405-419. doi:10.1007/s00521-009-0307-6*

    :param SpikeTrain a: First spike train.
    :param SpikeTrain b: Second spike train.
    :param kernel: Kernel to use. It corresponds to a smoothing filter
        by being the autocorrelation of such a filter.
    :type kernel: :class:`.signal_processing.Kernel`
    :returns: The Schreiber et al. similarity measure of the spike trains given
        the kernel.
    :rtype: float
    """

    k_dist = kernel.summed_dist_matrix(trains)
    vr_dist = sp.empty(k_dist.shape)
    for i, j in sp.ndindex(*k_dist.shape):
        if k_dist[i, i] == 0.0 or k_dist[j, j] == 0.0:
            vr_dist[i, j] = sp.nan
        else:
            vr_dist[i, j] = sp.sqrt(
                k_dist[i, j] * k_dist[j, i] / k_dist[i, i] / k_dist[j, j])
    return vr_dist
