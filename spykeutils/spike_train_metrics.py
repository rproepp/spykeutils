
from monkeypatch import quantities_patch
import heapq
import quantities as pq
import scipy as sp
import _scipy_quantities as spq
import signal_processing as sigproc
import tools

assert quantities_patch  # Suppress pyflakes warning, patch applied by loading


def _calc_multiunit_dist_matrix_from_single_trials(units, dist_func, **params):
    if len(units) <= 0:
        return sp.zeros((0, 0))

    num_trials = len(units.itervalues().next())
    if not all((len(v) == num_trials for v in units.itervalues())):
        raise ValueError("Number of trials differs among units.")

    D = sp.empty((num_trials, num_trials))
    for i in xrange(num_trials):
        D[i, i] = 0.0
        a = [units[k][i] for k in units.iterkeys()]
        for j in xrange(i + 1, num_trials):
            b = [units[k][j] for k in units.iterkeys()]
            D[i, j] = D[j, i] = dist_func(a, b, **params)
    return D


def _create_matrix_from_indexed_function(
        shape, func, symmetric_2d=False, **func_params):
    mat = sp.empty(shape)
    if symmetric_2d:
        for i in xrange(shape[0]):
            for j in xrange(i, shape[1]):
                mat[i, j] = mat[j, i] = func(i, j, **func_params)
    else:
        for idx in sp.ndindex(*shape):
            mat[idx] = func(*idx, **func_params)
    return mat


def _merge_trains_and_label_spikes(trains):
    labeled_trains = (zip(st, len(st) * (label,)) for label, st
                      in enumerate(trains))
    return list(heapq.merge(*labeled_trains))


def cs_dist(
        trains, smoothing_filter, sampling_rate,
        filter_area_fraction=sigproc.default_kernel_area_fraction):
    """ Calculates the Cauchy-Schwarz distance between two spike trains given
    a smoothing filter.

    Let :math:`v_a(t)` and :math:`v_b(t)` with :math:`t \\in \\mathcal{T}` be
    the spike trains convolved with some smoothing filter and :math:`V(a, b)
    = \\int_{\\mathcal{T}} v_a(t) v_b(t) dt`. Then, the Cauchy-Schwarz distance
    of the spike trains is defined as :math:`d_{CS}(a, b) = \\arccos \\frac{V(a,
    b)^2}{V(a, a) V(b, b)}`.

    The Cauchy-Schwarz distance is closely related to the Schreiber et al.
    similarity measure :math:`S_S` by :math:`d_{CS} = \\arccos S_S^2`

    This function numerically convolves the spike trains with the smoothing
    filter which can be quite slow and inaccurate. If the analytical result of
    the autocorrelation of the smoothing filter is known, one can use
    :func:`schreiber_similarity` for a more efficient and precise calculation.

    Further information can be found in *Paiva, A. R. C., Park, I., & Principe,
    J. (2010). Inner products for representation and learning in the spike
    train domain. Statistical Signal Processing for Neuroscience and
    Neurotechnology, Academic Press, New York.*

    :param sequence trains: Sequence of `SpikeTrain` of which the distance
        will be calculated pairwise.
    :param smoothing_filter: Smoothing filter to be convolved with the spike
        trains.
    :type smoothing_filter: :class:`.signal_processing.Kernel`
    :param sampling_rate: The sampling rate which will be used to bin
        the spike trains.
    :type sampling_rate: Quantity scalar
    :param float filter_area_fraction: A value between 0 and 1 which controls
        the interval over which the smoothing filter will be discretized. At
        least the given fraction of the complete smoothing filter area will be
        covered. Higher values can lead to more accurate results (besides the
        sampling rate).
    :returns: Matrix containing the Cauchy-Schwarz distance of all pairs of
        spike trains
    :rtype: 2-D array
    """

    inner = st_inner(
        trains, trains, smoothing_filter, sampling_rate, filter_area_fraction)
    return sp.arccos(
        inner ** 2 / sp.diag(inner) / sp.atleast_2d(sp.diag(inner)).T)


def event_synchronization(
        trains, tau=None, kernel=sigproc.RectangularKernel(1, normalize=False),
        sort=True):
    """ Calculates the event synchronization.

    Let :math:`d(x|y)` be the count of spikes in :math:`y` which occur shortly
    before an event in :math:`x` with a time difference of less then
    :math:`\\tau`. Moreover, let :math:`n_x` and :math:`n_y` be the number of
    total spikes in the spike trains :math:`x` and :math:`y`. The event
    synchrony is then defined as :math:`Q_T = \\frac{d(x|y)
    + d(y|x)}{\\sqrt{n_x n_y}}`.

    The time maximum time lag :math:`\\tau` can be determined automatically for
    each pair of spikes :math:`t^x_i` and :math:`t^y_j` by the formula
    :math:`\\tau_{ij} = \\frac{1}{2} \\min\{t^x_{i+1} - t^x_i, t^x_i - t^x_{i-1},
    t^y_{j+1} - t^y_j, t^y_j - t^y_{j-1}\}`

    Further and more detailed information can be found in
    *Quiroga, R. Q., Kreuz, T., & Grassberger, P. (2002). Event
    synchronization: a simple and fast method to measure synchronicity and time
    delay patterns. Physical Review E, 66(4), 041904.*

    :param sequence trains: SpikeTrains of which the van Rossum distance will be
        calculated pairwise.
    :param tau: The maximum time lag for two spikes to be considered coincident
        or synchronous. To have it determined automatically by above formula set
        it to `None`.
    :type tau: Quantity scalar
    :param kernel: Kernel to use in the calculation of the distance.
    :type kernel: :class:`.signal_processing.Kernel`
    :param bool sort: Spike trains with sorted spike times are be needed for
        the calculation. You can set `sort` to `False` if you know that your
        spike trains are already sorted to decrease calculation time.
    :returns: Matrix containing the event synchronization for all pairs of spike
        trains.
    :rtype: 2-D array
    """

    trains = [st.view(type=pq.Quantity) for st in trains]
    if sort:
        trains = [sp.sort(st) for st in trains]

    if tau is None:
        inf_array = sp.array([sp.inf])
        isis = [spq.concatenate(
                (inf_array * st.units, sp.diff(st), inf_array * st.units))
                for st in trains]
        auto_taus = [spq.minimum(t[:-1], t[1:]) for t in isis]

    def compute(i, j):
        if i == j:
            return 1.0
        else:
            if tau is None:
                tau_mat = spq.minimum(*spq.meshgrid(
                    auto_taus[i], auto_taus[j])) / 2.0
            else:
                tau_mat = sp.tile(tau, (trains[j].size, trains[i].size))
            coincidence = sp.sum(kernel(
                (trains[i] - sp.atleast_2d(trains[j]).T) / tau_mat))
            normalization = 1.0 / sp.sqrt(trains[i].size * trains[j].size)
            return normalization * coincidence

    return _create_matrix_from_indexed_function(
        (len(trains), len(trains)), compute, kernel.is_symmetric())


def hunter_milton_similarity(trains, tau=1.0 * pq.s, kernel=None):
    """ Calculates the Hunter-Milton similarity measure.

    If the kernel function is denoted as :math:`K(t)`, a function :math:`d(x_k)
    = K(x_k - y_{k'})` can be defined with :math:`y_{k'}` being the closest
    spike in spike train :math:`y` to the spike :math:`x_k` in spike train
    :math:`x`. With this the Hunter-Milton similarity measure is :math:`S_H =
    \\frac{1}{2} \\left(\\frac{1}{n_x} \\sum_{k = 1}^{n_x} d(x_k)
    + \\frac{1}{n_y} \\sum_{k' = 1}^{n_y} d(y_{k'})\\right)`.

    This implementation returns 0 if one of the spike trains is empty, but 1 if
    both are empty.

    Further information can be found in

    - *Hunter, J. D., & Milton, J. G. (2003). Amplitude and Frequency
      Dependence of Spike Timing: Implications for Dynamic Regulation. Journal
      of Neurophysiology.*
    - *Dauwels, J., Vialatte, F., Weber, T., & Cichocki, A. (2009). On
      similarity measures for spike trains. Advances in Neuro-Information
      Processing, 177-185.*

    :param sequence trains: SpikeTrains of which the Hunter-Milton similarity
        will be calculated pairwise.
    :param tau: The time scale for determining the coincidence of two events.
    :type tau: Quantity scalar
    :param kernel: Kernel to use in the calculation of the distance. If `None`,
        a unnormalized Laplacian kernel will be used.
    :type kernel: :class:`.signal_processing.Kernel`
    :returns: Matrix containing the Hunter-Milton similarity for all pairs of
        spike trains.
    :rtype: 2-D array
    """

    if kernel is None:
        kernel = sigproc.LaplacianKernel(tau, normalize=False)

    def compute(i, j):
        if i == j:
            return 1.0
        elif trains[i].size <= 0 or trains[j].size <= 0:
            return 0.0
        else:
            diff_matrix = sp.absolute(trains[i] - sp.atleast_2d(trains[j]).T)
            return 0.5 * (
                sp.sum(kernel(sp.amin(diff_matrix, axis=0))) / trains[i].size +
                sp.sum(kernel(sp.amin(diff_matrix, axis=1))) / trains[j].size)

    return _create_matrix_from_indexed_function(
        (len(trains), len(trains)), compute, kernel.is_symmetric())


def norm_dist(
        trains, smoothing_filter, sampling_rate,
        filter_area_fraction=sigproc.default_kernel_area_fraction):
    """ Calculates the norm distance between spike trains given a smoothing
    filter.

    Let :math:`v_a(t)` and :math:`v_b(t)` with :math:`t \\in \\mathcal{T}` be
    the spike trains convolved with some smoothing filter. Then, the norm
    distance of the spike trains is defined as :math:`d_{ND}(a, b)
    = \\sqrt{\\int_{\\mathcal{T}} (v_a(t) - v_b(t))^2 dt}`.

    Further information can be found in *Paiva, A. R. C., Park, I., & Principe,
    J. (2010). Inner products for representation and learning in the spike
    train domain. Statistical Signal Processing for Neuroscience and
    Neurotechnology, Academic Press, New York.*

    :param sequence trains: Sequence of `SpikeTrain` of which the distance
        will be calculated pairwise.
    :param SpikeTrain b: Second spike train.
    :param smoothing_filter: Smoothing filter to be convolved with the spike
        trains.
    :type smoothing_filter: :class:`.signal_processing.Kernel`
    :param sampling_rate: The sampling rate which will be used to bin
        the spike trains.
    :type sampling_rate: Quantity scalar
    :param float filter_area_fraction: A value between 0 and 1 which controls
        the interval over which the smoothing filter will be discretized. At
        least the given fraction of the complete smoothing filter area will be
        covered. Higher values can lead to more accurate results (besides the
        sampling rate).
    :returns: Matrix containing the norm distance of all pairs of spike trains
        given the smoothing_filter.
    :rtype: Quantity 2D
    """

    inner = st_inner(
        trains, trains, smoothing_filter, sampling_rate, filter_area_fraction)
    return spq.maximum(
        0.0 * pq.Hz,
        (spq.diag(inner) + sp.atleast_2d(spq.diag(inner)).T - 2 * inner)) ** 0.5


def schreiber_similarity(trains, kernel, sort=True):
    """ Calculates the Schreiber et al. similarity measure between spike
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

    :param sequence trains: Sequence of `SpikeTrain` of which the distance
        will be calculated pairwise.
    :param kernel: Kernel to use. It corresponds to a smoothing filter
        by being the autocorrelation of such a filter.
    :type kernel: :class:`.signal_processing.Kernel`
    :param bool sort: Spike trains with sorted spike times will be needed for
        the calculation. You can set `sort` to `False` if you know that your
        spike trains are already sorted to decrease calculation time.
    :returns: Matrix containing the Schreiber et al. similarity measure of all
        pairs of spike trains.
    :rtype: 2-D array
    """

    k_dist = kernel.summed_dist_matrix(trains, not sort)

    def compute(i, j):
        return sp.sqrt(
            k_dist[i, j] * k_dist[j, i] / k_dist[i, i] / k_dist[j, j])

    return _create_matrix_from_indexed_function(
        (len(trains), len(trains)), compute, kernel.is_symmetric())


def st_inner(
        a, b, smoothing_filter, sampling_rate,
        filter_area_fraction=sigproc.default_kernel_area_fraction):
    """ Calculates the inner product of spike trains given a smoothing
    filter.

    Let :math:`v_a(t)` and :math:`v_b(t)` with :math:`t \\in \\mathcal{T}` be
    the spike trains convolved with some smoothing filter. Then, the inner
    product of the spike trains is defined as :math:`\\int_{\\mathcal{T}}
    v_a(t)v_b(t) dt`.

    Further information can be found in *Paiva, A. R. C., Park, I., & Principe,
    J. (2010). Inner products for representation and learning in the spike
    train domain. Statistical Signal Processing for Neuroscience and
    Neurotechnology, Academic Press, New York.*

    :param sequence a: Sequence of `SpikeTrain`.
    :param sequence b: Sequence of `SpikeTrain`.
    :param smoothing_filter: A smoothing filter to be convolved with the spike
        trains.
    :type smoothing_filter: :class:`.signal_processing.Kernel`
    :param sampling_rate: The sampling rate which will be used to bin
        the spike train.
    :type sampling_rate: Quantity scalar
    :param float filter_area_fraction: A value between 0 and 1 which controls
        the interval over which the `smoothing_filter` will be discretized. At
        least the given fraction of the complete `smoothing_filter` area will be
        covered. Higher values can lead to more accurate results (besides the
        sampling rate).
    :returns: Matrix containing the inner product for each pair of spike trains
        with one spike train from `a` and the other one from `b`.
    :rtype: Quantity 2D
    """

    if all((x is y for x, y in zip(a, b))):
        convolved, sampling_rate = _prepare_for_inner_prod(
            a, smoothing_filter, sampling_rate, filter_area_fraction)
        convolved = convolved + convolved
    else:
        convolved, sampling_rate = _prepare_for_inner_prod(
            a + b, smoothing_filter, sampling_rate, filter_area_fraction)
    return (sp.inner(convolved[:len(a)], convolved[len(a):]) *
            convolved[0].units * convolved[1].units / sampling_rate)


def _prepare_for_inner_prod(
        trains, smoothing_filter, sampling_rate, filter_area_fraction):
    t_start, t_stop = tools.maximum_spike_train_interval({0: trains})
    padding = smoothing_filter.boundary_enclosing_at_least(filter_area_fraction)
    t_start -= 2 * padding
    t_stop += 2 * padding

    return [sigproc.st_convolve(
        st, smoothing_filter, sampling_rate, mode='full',
        binning_params={'t_start': t_start, 't_stop': t_stop},
        kernel_discretization_params={'area_fraction': filter_area_fraction})[0]
        for st in trains], sampling_rate


def st_norm(
        train, smoothing_filter, sampling_rate,
        filter_area_fraction=sigproc.default_kernel_area_fraction):
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
    :param sampling_rate: The sampling rate which will be used to bin
        the spike train.
    :type sampling_rate: Quantity scalar
    :param float filter_area_fraction: A value between 0 and 1 which controls
        the interval over which the smoothing filter will be discretized. At
        least the given fraction of the complete smoothing filter area will be
        covered. Higher values can lead to more accurate results (besides the
        sampling rate).
    :returns: The of norm the spike train given the smoothing_filter.
    :rtype: Quantity scalar
    """

    return st_inner(
        [train], [train], smoothing_filter, sampling_rate,
        filter_area_fraction) ** 0.5


def van_rossum_dist(trains, tau=1.0 * pq.s, kernel=None, sort=True):
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
    kernels usually have run-time complexity of :math:`O(N^2 n^2)` depending on
    the implementation of :meth:`.Kernel.summed_dist_matrix`.

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
    :param bool sort: Spike trains with sorted spike times might be needed for
        the calculation. You can set `sort` to `False` if you know that your
        spike trains are already sorted to decrease calculation time.
    :returns: Matrix containing the van Rossum distances for all pairs of spike
        trains.
    :rtype: 2-D array
    """

    if kernel is None:
        if tau == sp.inf:
            spike_counts = [st.size for st in trains]
            return sp.absolute(spike_counts - sp.atleast_2d(spike_counts).T)
        kernel = sigproc.LaplacianKernel(tau, normalize=False)

    k_dist = kernel.summed_dist_matrix(trains, not sort)
    vr_dist = sp.empty_like(k_dist)
    for i, j in sp.ndindex(*k_dist.shape):
        vr_dist[i, j] = (
            k_dist[i, i] + k_dist[j, j] - k_dist[i, j] - k_dist[j, i])
    return sp.sqrt(vr_dist)


def van_rossum_multiunit_dist(units, weighting, tau=1.0 * pq.s, kernel=None):
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

    :param dict units: Dictionary of lists with each list containing the trials
            of one unit. Each trial should be a `SpikeTrain` and all units
            should have the same number of trials.
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
    :returns: A 2D array with the multi-unit distance for each pair of trials.
    :rtype: 2D arrary
    """

    if kernel is None and tau != sp.inf:
        kernel = sigproc.LaplacianKernel(tau, normalize=False)

    return _calc_multiunit_dist_matrix_from_single_trials(
        units, _van_rossum_multiunit_dist_for_trial_pair, weighting=weighting,
        tau=tau, kernel=kernel)


def _van_rossum_multiunit_dist_for_trial_pair(a, b, weighting, tau, kernel):
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


def victor_purpura_dist(trains, q=1.0 * pq.Hz, kernel=None, sort=True):
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

    Given the average number of spikes :math:`n` in a spike train and :math:`N`
    spike trains the run-time complexity of this function is
    :math:`O(N^2 n^2)` and :math:`O(N^2 + n^2)` memory will be needed.

    :param sequence trains: Sequence of `SpikeTrain` of which the distance
        will be calculated pairwise.
    :param q: Cost factor for spike shifts. If `kernel` is not `None`, `q` will
        be ignored.
    :type q: Quantity scalar
    :param kernel: Kernel to use in the calculation of the distance. If
        `kernel` is `None`, an unnormalized triangular kernel with a half width
        of `2.0/q` will be used.
    :type kernel: :class:`.signal_processing.Kernel`
    :param bool sort: Spike trains with sorted spike times will be needed for
        the calculation. You can set `sort` to `False` if you know that your
        spike trains are already sorted to decrease calculation time.
    :returns: Matrix containing the VP distance of all pairs of spike trains.
    :rtype: 2-D array
    """

    if kernel is None:
        kernel = sigproc.TriangularKernel(2.0 / q, normalize=False)

    if sort:
        trains = [sp.sort(st.view(type=pq.Quantity)) for st in trains]

    def compute(i, j):
        if i == j:
            return 0.0
        else:
            return _victor_purpura_dist_for_trial_pair(
                trains[i], trains[j], kernel)

    return _create_matrix_from_indexed_function(
        (len(trains), len(trains)), compute, kernel.is_symmetric())


#@profile
def _victor_purpura_dist_for_trial_pair(a, b, kernel):
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
    # It constructs a matrix G[i, j] containing the minimal cost when only
    # considering the first i and j spikes of the spike trains. However, one
    # never needs to store more than one row and one column at the same time
    # for calculating the VP distance.
    # cost[0, :cost.shape[1] - num_spikes_processed] corresponds to
    # G[num_spikes_processed:, num_spikes_processed]. In the same way
    # cost[1, :cost.shape[1] - num_spikes_processed] corresponds to
    # G[num_spikes_processed, num_spikes_processed:].
    #
    # Moreover, the minimum operation on the costs of the three kind of actions
    # (delete, insert or move spike) can be split up in two operations. One
    # operation depends only on the already calculated costs and kernel
    # evaluation (insertion of spike vs moving a spike). The other minimum
    # depends on that result and the cost of deleting a spike. This operation
    # always depends on the last calculated element in the cost array and
    # corresponds to a recursive application of
    # f(x[i]) = min(f(x[i-1]), x[i]) + 1. That '+1' can be excluded from this
    # function if the summed value for all recursive applications is added
    # upfront to x. Afterwards it has to be removed again except one for the
    # currently processed spike to get the real costs up to the evaluation of
    # num_spikes_processed. One can save here a couple of additions by
    # intelligently shifting the min_summands array.

    cost = sp.asfortranarray(sp.tile(sp.arange(a.size + 1.0), (2, 1)))
    min_summands = sp.asfortranarray(cost[:, ::-1])
    k = 1 - 2 * sp.asfortranarray(kernel(
        (sp.atleast_2d(a).T - b).view(type=pq.Quantity)).simplified)

    for num_spikes_processed in xrange(b.size):
        x = sp.minimum(
            cost[:, 1:cost.shape[1] - num_spikes_processed],
            cost[:, :-num_spikes_processed - 1] +
            k[num_spikes_processed:, num_spikes_processed])
        x[:, 0] = min(cost[1, 1], x[0, 0])
        x += min_summands[:, -x.shape[1] - 1:-1]
        x = sp.minimum.accumulate(x, axis=1)
        cost[:, :x.shape[1]] = x - min_summands[:, -x.shape[1]:]

    return cost[0, -b.size - 1]


def victor_purpura_multiunit_dist(
        units, reassignment_cost, q=1.0 * pq.Hz, kernel=None):
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
    units with :math:`n` spike trains each the run-time complexity is
    :math:`O(n^2 LN^{L+1})`. The space complexity is :math:`O(n^2 + LN^{L+1}`.

    For calculating the distance between only two units one should use
    :func:`victor_purpura_dist` which is more memory efficient.

    :param dict units: Dictionary of lists with each list containing the trials
            of one unit. Each trial should be a `SpikeTrain` and all units
            should have the same number of trials.
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
    :returns: A 2D array with the multi-unit distance for each pair of trials.
    :rtype: 2D arrary
    """

    if kernel is None:
        kernel = sigproc.TriangularKernel(2.0 / q, normalize=False)
    return _calc_multiunit_dist_matrix_from_single_trials(
        units, _victor_purpura_multiunit_dist_for_trial_pair,
        reassignment_cost=reassignment_cost, kernel=kernel)


def _victor_purpura_multiunit_dist_for_trial_pair(
        a, b, reassignment_cost, kernel):
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
                cost_delete_in_a, cost_delete_in_b, cost_shift)

    return cost.flat[-1]
