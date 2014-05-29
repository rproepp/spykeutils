from monkeypatch import quantities_patch
import quantities as pq
import scipy as sp
import _scipy_quantities as spq
import signal_processing as sigproc
import tools

try:
    import pymuvr
    PYMUVR_AVAILABLE = True
except ImportError:
    PYMUVR_AVAILABLE = False

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
    labels = sp.concatenate(
        [sp.zeros(st.size, dtype=int) + i for i, st in enumerate(trains)])
    trains = spq.concatenate([st.view(dtype=pq.Quantity) for st in trains])
    sorted_indices = sp.argsort(trains)
    return trains[sorted_indices], labels[sorted_indices]


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

    :param sequence trains: Sequence of :class:`neo.core.SpikeTrain` objects of
        which the distance will be calculated pairwise.
    :param smoothing_filter: Smoothing filter to be convolved with the spike
        trains.
    :type smoothing_filter: :class:`.signal_processing.Kernel`
    :param sampling_rate: The sampling rate which will be used to bin
        the spike trains as inverse time scalar.
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
        trains, tau=None,
        kernel=sigproc.RectangularKernel(1.0, normalize=False), sort=True):
    """ event_synchronization(trains, tau=None, kernel=signal_processing.RectangularKernel(1.0, normalize=False), sort=True)

    Calculates the event synchronization.

    Let :math:`d(x|y)` be the count of spikes in :math:`y` which occur shortly
    before an event in :math:`x` with a time difference of less than
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

    :param sequence trains: Sequence of :class:`neo.core.SpikeTrain` objects of
        which the van Rossum distance will be calculated pairwise.
    :param tau: The maximum time lag for two spikes to be considered coincident
        or synchronous as time scalar. To have it determined automatically by
        above formula set it to `None`.
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

    :param sequence trains: Sequence of :class:`neo.core.SpikeTrain` objects of
        which the Hunter-Milton similarity will be calculated pairwise.
    :param tau: The time scale for determining the coincidence of two events as
        time scalar.
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

    :param sequence trains: Sequence of :class:`neo.core.SpikeTrain` objects of
        which the distance will be calculated pairwise.
    :param smoothing_filter: Smoothing filter to be convolved with the spike
        trains.
    :type smoothing_filter: :class:`.signal_processing.Kernel`
    :param sampling_rate: The sampling rate which will be used to bin
        the spike trains as inverse time scalar.
    :type sampling_rate: Quantity scalar
    :param float filter_area_fraction: A value between 0 and 1 which controls
        the interval over which the smoothing filter will be discretized. At
        least the given fraction of the complete smoothing filter area will be
        covered. Higher values can lead to more accurate results (besides the
        sampling rate).
    :returns: Matrix containing the norm distance of all pairs of spike trains
        given the smoothing_filter.
    :rtype: Quantity 2D with units depending on the smoothing filter (usually
        temporal frequency units)
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

    :param sequence trains: Sequence of :class:`neo.core.SpikeTrain` objects of
        which the distance will be calculated pairwise.
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

    :param sequence a: Sequence of :class:`neo.core.SpikeTrain` objects.
    :param sequence b: Sequence of :class:`neo.core.SpikeTrain` objects.
    :param smoothing_filter: A smoothing filter to be convolved with the spike
        trains.
    :type smoothing_filter: :class:`.signal_processing.Kernel`
    :param sampling_rate: The sampling rate which will be used to bin
        the spike train as inverse time scalar.
    :type sampling_rate: Quantity scalar
    :param float filter_area_fraction: A value between 0 and 1 which controls
        the interval over which the `smoothing_filter` will be discretized. At
        least the given fraction of the complete `smoothing_filter` area will be
        covered. Higher values can lead to more accurate results (besides the
        sampling rate).
    :returns: Matrix containing the inner product for each pair of spike trains
        with one spike train from `a` and the other one from `b`.
    :rtype: Quantity 2D with units depending on the smoothing filter (usually
        temporal frequency units)
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

    :param train: Spike train of which to calculate the norm.
    :type train: :class:`neo.core.SpikeTrain`
    :param smoothing_filter: Smoothing filter to be convolved with the spike
        train.
    :type smoothing_filter: :class:`.signal_processing.Kernel`
    :param sampling_rate: The sampling rate which will be used to bin
        the spike train as inverse time scalar.
    :type sampling_rate: Quantity scalar
    :param float filter_area_fraction: A value between 0 and 1 which controls
        the interval over which the smoothing filter will be discretized. At
        least the given fraction of the complete smoothing filter area will be
        covered. Higher values can lead to more accurate results (besides the
        sampling rate).
    :returns: The norm of the spike train given the smoothing_filter.
    :rtype: Quantity scalar with units depending on the smoothing filter (usually
        temporal frequency units)
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
    complexity of this function is :math:`O(N^2 n^2)`. An implementation in
    :math:`O(N^2 n)` would be possible but has a high constant factor rendering
    it slower in practical cases.

    :param sequence trains: Sequence of :class:`neo.core.SpikeTrain` objects of
        which the van Rossum distance will be calculated pairwise.
    :param tau: Decay rate of the exponential function as time scalar. Controls
        for which time scale the metric will be sensitive. This parameter will
        be ignored if `kernel` is not `None`. May also be :const:`scipy.inf`
        which will lead to only measuring differences in spike count.
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
            return (spike_counts - sp.atleast_2d(spike_counts).T) ** 2
        kernel = sigproc.LaplacianKernel(tau, normalize=False)

    k_dist = kernel.summed_dist_matrix(
        [st.view(type=pq.Quantity) for st in trains], not sort)
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
    run-time complexity of this function is :math:`O(N^2 n^2)` and :math:`O(N^2
    + Nn^2)` memory will be needed.

    If `pymuvr` is installed, this function will use the faster C++
    implementation contained in the package.

    :param dict units: Dictionary of sequences with each sequence containing
        the trials of one unit. Each trial should be
        a :class:`neo.core.SpikeTrain` and all units should have the same
        number of trials.
    :param float weighting: Controls the interpolation between a labeled line
        and a summed population coding.
    :param tau: Decay rate of the exponential function as time scalar. Controls
        for which time scale the metric will be sensitive. This parameter will
        be ignored if `kernel` is not `None`. May also be :const:`scipy.inf`
        which will lead to only measuring differences in spike count.
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

    if PYMUVR_AVAILABLE and tau != sp.inf:
        rescaled_trains = []
        n_trials = len(units.itervalues().next())

        for i in xrange(n_trials):
            trial_trains = []
            for u, tr in units.iteritems():
                trial_trains.append(list(tr[i].rescale(pq.s).magnitude))
            rescaled_trains.append(trial_trains)
        t = tau.rescale(pq.s).magnitude
        r = pymuvr.square_distance_matrix(rescaled_trains, weighting, t)
        print r
        #print rescaled_trains, weighting, t
        print _calc_multiunit_dist_matrix_from_single_trials(
            units, _van_rossum_multiunit_dist_for_trial_pair, weighting=weighting,
            tau=tau, kernel=kernel)
        return r

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

    :param sequence trains: Sequence of :class:`neo.core.SpikeTrain` objects of
        which the distance will be calculated pairwise.
    :param q: Cost factor for spike shifts as inverse time scalar. If `kernel`
        is not `None`, `q` will be ignored.
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
        if q == 0.0:
            num_spikes = sp.atleast_2d([st.size for st in trains])
            return sp.absolute(num_spikes.T - num_spikes)
        else:
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
    # cost[0, :cost.shape[1] - i] corresponds to G[i:, i]. In the same way
    # cost[1, :cost.shape[1] - i] corresponds to G[i, i:].
    #
    # Moreover, the minimum operation on the costs of the three kind of actions
    # (delete, insert or move spike) can be split up in two operations. One
    # operation depends only on the already calculated costs and kernel
    # evaluation (insertion of spike vs moving a spike). The other minimum
    # depends on that result and the cost of deleting a spike. This operation
    # always depends on the last calculated element in the cost array and
    # corresponds to a recursive application of
    # f(accumulated_min[i]) = min(f(accumulated_min[i-1]), accumulated_min[i])
    # + 1. That '+1' can be excluded from this function if the summed value for
    # all recursive applications is added upfront to accumulated_min.
    # Afterwards it has to be removed again except one for the currently
    # processed spike to get the real costs up to the evaluation of i.
    #
    # All currently calculated costs will be considered -1 because this saves
    # a number of additions as in most cases the cost would be increased by
    # exactly one (the only exception is shifting, but in that calculation is
    # already the addition of a constant involved, thus leaving the number of
    # operations the same). The increase by one will be added after calculating
    # all minima by shifting decreasing_sequence by one when removing it from
    # accumulated_min.

    min_dim, max_dim = b.size, a.size + 1
    cost = sp.asfortranarray(sp.tile(sp.arange(float(max_dim)), (2, 1)))
    decreasing_sequence = sp.asfortranarray(cost[:, ::-1])
    k = 1 - 2 * sp.asfortranarray(kernel(
        (sp.atleast_2d(a).T - b).view(type=pq.Quantity)).simplified)

    for i in xrange(min_dim):
        # determine G[i, i] == accumulated_min[:, 0]
        #accumulated_min = sp.empty((2, max_dim - i - 1))
        accumulated_min = cost[:, :-i - 1] + k[i:, i]
        accumulated_min[1, :b.size - i] = cost[1, :b.size - i] + k[i, i:]
        accumulated_min = sp.minimum(
            accumulated_min,  # shift
            cost[:, 1:max_dim - i])  # insert
        acc_dim = accumulated_min.shape[1]
        # delete vs min(insert, shift)
        accumulated_min[:, 0] = min(cost[1, 1], accumulated_min[0, 0])
        # determine G[i, :] and G[:, i] by propagating minima.
        accumulated_min += decreasing_sequence[:, -acc_dim - 1:-1]
        accumulated_min = sp.minimum.accumulate(accumulated_min, axis=1)
        cost[:, :acc_dim] = accumulated_min - decreasing_sequence[:, -acc_dim:]

    return cost[0, -min_dim - 1]


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
    :math:`O(n^2 LN^{L+1})`. The space complexity is :math:`O(n^2 + LN^{L+1})`.

    For calculating the distance between only two units one should use
    :func:`victor_purpura_dist` which is more efficient.

    :param dict units: Dictionary of sequences with each sequence containing
        the trials of one unit. Each trial should be
        a :class:`neo.core.SpikeTrain` and all units should have the same
        number of trials.
    :param float reassignment_cost: Cost to reassign a spike from one train to
        another (sometimes denoted with :math:`k`). Should be between 0 and 2.
        For 0 spikes can be reassigned without any cost, for 2 and above it is
        cheaper to delete and reinsert a spike.
    :param q: Cost factor for spike shifts as inverse time scalar. If `kernel`
        is not `None`, `q` will be ignored.
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
    # this matrix with G). In this implementation the only the one submatrix
    # for one specific i is stored as in each step only i-1 and i will be
    # accessed. That saves some memory.

    # Initialization of various variables needed by the algorithm. Also swap
    # a and b if it will save time as the algorithm is not symmetric.
    a_num_spikes = [st.size for st in a]
    b_num_spikes = [st.size for st in b]
    a_num_total_spikes = sp.sum(a_num_spikes)

    complexity_same = a_num_total_spikes * sp.prod(b_num_spikes)
    complexity_swapped = sp.prod(a_num_spikes) * sp.sum(b_num_spikes)
    if complexity_swapped < complexity_same:
        a, b = b, a
        a_num_spikes, b_num_spikes = b_num_spikes, a_num_spikes
        a_num_total_spikes = sp.sum(a_num_spikes)

    if a_num_total_spikes <= 0:
        return sp.sum(b_num_spikes)

    b_dims = tuple(sp.asarray(b_num_spikes) + 1)

    cost = sp.asfarray(sp.sum(sp.indices(b_dims), axis=0))

    a_merged = _merge_trains_and_label_spikes(a)
    b_strides = sp.cumprod((b_dims + (1,))[::-1])[:-1]
    flat_b_indices = sp.arange(cost.size)
    b_indices = sp.vstack(sp.unravel_index(flat_b_indices, b_dims))
    flat_neighbor_indices = sp.maximum(
        0, sp.atleast_2d(flat_b_indices).T - b_strides[::-1])
    invalid_neighbors = b_indices.T == 0

    b_train_mat = sp.empty((len(b), sp.amax(b_num_spikes))) * b[0].units
    for i, st in enumerate(b):
        b_train_mat[i, :st.size] = st.rescale(b[0].units)
        b_train_mat[i, st.size:] = sp.nan * b[0].units

    reassignment_costs = sp.empty((a_merged[0].size,) + b_train_mat.shape)
    reassignment_costs.fill(reassignment_cost)
    reassignment_costs[sp.arange(a_merged[1].size), a_merged[1], :] = 0.0
    k = 1 - 2 * kernel(sp.atleast_2d(
        a_merged[0]).T - b_train_mat.flatten()).simplified.reshape(
            (a_merged[0].size,) + b_train_mat.shape) + reassignment_costs

    decreasing_sequence = flat_b_indices[::-1]

    # Do the actual calculations.
    for a_idx in xrange(1, a_num_total_spikes + 1):
        base_costs = cost.flat[flat_neighbor_indices]
        base_costs[invalid_neighbors] = sp.inf
        min_base_cost_labels = sp.argmin(base_costs, axis=1)
        cost_all_possible_shifts = k[a_idx - 1, min_base_cost_labels, :] + \
            sp.atleast_2d(base_costs[flat_b_indices, min_base_cost_labels]).T
        cost_shift = cost_all_possible_shifts[
            sp.arange(cost_all_possible_shifts.shape[0]),
            b_indices[min_base_cost_labels, flat_b_indices] - 1]

        cost_delete_in_a = cost.flat[flat_b_indices]

        # cost_shift is dimensionless, but there is a bug in quantities with
        # the minimum function:
        # <https://github.com/python-quantities/python-quantities/issues/52>
        # The explicit request for the magnitude circumvents this problem.
        cost.flat = sp.minimum(cost_delete_in_a, cost_shift.magnitude) + 1
        cost.flat[0] = sp.inf

        # Minimum with cost for deleting in b
        # The calculation order is somewhat different from the order one would
        # expect from the naive algorithm. This implementation, however,
        # optimizes the use of the CPU cache giving a considerable speed
        # improvement.
        # Basically this codes calculates the values of a row of elements for
        # each dimension of cost.
        for dim_size, stride in zip(b_dims[::-1], b_strides):
            for i in xrange(stride):
                segment_size = dim_size * stride
                for j in xrange(i, cost.size, segment_size):
                    s = sp.s_[j:j + segment_size:stride]
                    seq = decreasing_sequence[-cost.flat[s].size:]
                    cost.flat[s] = sp.minimum.accumulate(
                        cost.flat[s] + seq) - seq

    return cost.flat[-1]
