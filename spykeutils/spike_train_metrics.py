
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

    k_dist = sigproc.LaplacianKernel(tau).summed_dist_matrix(trains)
    vr_dist = sp.empty_like(k_dist)
    for i, j in sp.ndindex(*k_dist.shape):
        vr_dist[i, j] = (
            k_dist[i, i] + k_dist[j, j] - k_dist[i, j] - k_dist[j, i])
    return sp.sqrt(2.0 * tau * vr_dist)


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


def schreiber_similarity(a, b, kernel):
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
    :param function kernel: Kernel to use. It corresponds to a smoothing filter
        by being the autocorrelation of such a filter. The kernel function
        expects Quantity 1D array as argument denoting the time points for
        evaluation and should return a Quantity 1D array. The kernel has to be
        symmetric.
    :returns: The Schreiber et al. similarity measure of the spike trains given
        the kernel.
    :rtype: float
    """
    if a.size <= 0 or b.size <= 0:
        return sp.nan

    D = kernel.summed_dist_matrix([a, b])
    assert abs(D[1, 0] - D[0, 1]) < abs(1e-7 * D[1, 0])
    return D[0, 1] / ((D[0, 0] * D[1, 1]) ** 0.5)
