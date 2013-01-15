"""
.. data:: default_sampling_rate
"""

import quantities as pq
import scipy as sp
import scipy.signal
import scipy.special

default_kernel_area_fraction = 0.99999
default_sampling_rate = 1000 * pq.Hz


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


class Kernel(object):
    """ Base class for kernels. """

    def __init__(self, kernel_func, **params):
        """
        :param function kernel_func: Kernel function which takes an array
            of time points as first argument.
        :param dict params: Additional parameters to be passed to the kernel
            function.
        """
        self.kernel_func = kernel_func
        self.params = params

    def __call__(self, t):
        return self.kernel_func(t, **self.params)

    def boundary_enclosing_at_least(self, fraction):
        """ Calculates the boundary :math:`b` that the integral from :math:`-b`
        to :math:`b` encloses at least a certain fraction of the integral
        over the complete kernel.

        :param float fraction: Fraction of the whole area which at least has to
            be enclosed.
        :returns: boundary
        :rtype: Quantity scalar
        """
        raise NotImplementedError()

    def discretize(self, area_fraction, sampling_rate=default_sampling_rate):
        """ Discretizes the kernel.

        :param float area_fraction: Fraction between 0 and 1 (exclusive)
            of the integral of the kernel which will be at least covered by the
            discretization.
        :param sampling_rate: Sampling rate for the discretization.
        :type sampling_rate: Quantity scalar
        :rtype: Quantity 1D
        """

        t_step = 1.0 / sampling_rate
        boundary = self.boundary_enclosing_at_least(area_fraction)
        start = sp.ceil(-boundary / t_step)
        stop = sp.floor(boundary / t_step) + 1
        k = self(sp.arange(start, stop) * t_step)
        return k

    def summed_dist_matrix(self, vectors):
        """ Calculates the sum of all distances element pair distances for each
        pair of vectors.

        If :math:`(a_1, \\dots, a_n)` and :math:`(b_1, \\dots, b_m)` is a pair
        of vectors from `vectors` and :math:`K` the kernel, the resulting entry
        in the 2D array will be :math:`D_{ij} = \\sum_{i=1}^{n} \\sum_{j=1}^{m}
        K(a_i - b_j)`.

        :param sequence vectors: A sequence of 1D arrays to calculate the summed
            distances for each pair.
        :rtype: 2D array
        """

        D = sp.empty((len(vectors), len(vectors)))
        if len(vectors) > 0:
            might_have_units = self(vectors[0])
            if hasattr(might_have_units, 'units'):
                D = D * might_have_units.units

        for i, j in sp.ndindex(len(vectors), len(vectors)):
            D[i, j] = sp.sum(self(
                (vectors[i] - sp.atleast_2d(vectors[j]).T).flatten()))
        return D


class CausalDecayingExpKernel(Kernel):
    @staticmethod
    def evaluate(t, kernel_size):
        result = sp.piecewise(
            t, [t < 0, t >= 0], [
                lambda t: 0,
                lambda t: sp.exp((-t / kernel_size).simplified) / kernel_size])
        return result / kernel_size.units

    def __init__(self, kernel_size=1.0 * pq.s):
        Kernel.__init__(self, self.evaluate, kernel_size=kernel_size)

    def boundary_enclosing_at_least(self, fraction):
        return -self.params['kernel_size'] * sp.log(1.0 - fraction)


class GaussianKernel(Kernel):
    @staticmethod
    def evaluate(t, kernel_size):
        return (1.0 / (sp.sqrt(2 * sp.pi) * kernel_size) *
                sp.exp(-0.5 * (t / kernel_size).simplified ** 2))

    def __init__(self, kernel_size=1.0 * pq.s):
        Kernel.__init__(self, self.evaluate, kernel_size=kernel_size)

    def boundary_enclosing_at_least(self, fraction):
        return self.params['kernel_size'] * sp.sqrt(2.0) * \
            scipy.special.erfinv(fraction + scipy.special.erf(0.0))


class LaplacianKernel(Kernel):
    @staticmethod
    def evaluate(t, kernel_size):
        return sp.exp(-(sp.absolute(t) / kernel_size).simplified) \
            / (2.0 * kernel_size)

    def __init__(self, kernel_size=1.0 * pq.s):
        Kernel.__init__(self, self.evaluate, kernel_size=kernel_size)

    def boundary_enclosing_at_least(self, fraction):
        return -self.params['kernel_size'] * sp.log(1.0 - fraction)

    def summed_dist_matrix(self, vectors):
        # This implementation is based on
        #
        # Houghton, C., & Kreuz, T. (2012). On the efficient calculation of van
        # Rossum distances. Network: Computation in Neural Systems, 23(1-2),
        # 48-58.
        #
        # Note that the cited paper contains some errors.
        #
        # Given N vectors with n entries on average the run-time complexity is
        # O(N^2 * n). O(N^2 + N * n^2) memory will be needed.

        kernel_size = self.params['kernel_size']
        exp_vecs = [sp.exp(v / kernel_size) for v in vectors]
        inv_exp_vecs = [1.0 / v for v in exp_vecs]
        exp_diffs = [sp.outer(v, iv) for v, iv in zip(exp_vecs, inv_exp_vecs)]

        markage = [sp.empty(v.size) for v in vectors]
        for u in xrange(len(markage)):
            if markage[u].size <= 0:
                continue
            markage[u][0] = 0
            for i in xrange(1, markage[u].size):
                markage[u][i] = (
                    (markage[u][i - 1] + 1.0) * exp_diffs[u][i - 1, i])

        # Same vector terms
        D = sp.zeros((len(vectors), len(vectors)))
        for u in xrange(D.shape[0]):
            D[u, u] = markage[u].size + 2.0 * sp.sum(markage[u])

        # Cross vector terms
        for u in xrange(D.shape[0]):
            for v in xrange(u + 1, D.shape[1]):
                js, ks = _searchsorted_pairwise(vectors[u], vectors[v])
                start_j = sp.searchsorted(js, 0)
                start_k = sp.searchsorted(ks, 0)
                for i, j in enumerate(js[start_j:], start_j):
                    D[u, v] += (inv_exp_vecs[u][i] * exp_vecs[v][j] *
                                (1.0 + markage[v][j]))
                for i, k in enumerate(ks[start_k:], start_k):
                    D[u, v] += (inv_exp_vecs[v][i] * exp_vecs[u][k] *
                                (1.0 + markage[u][k]))
                D[v, u] = D[u, v]

        return D / 2.0 / kernel_size


class RectangularKernel(Kernel):
    @staticmethod
    def evaluate(t, half_width):
        return (sp.absolute(t) < half_width) / (2.0 * half_width)

    def __init__(self, half_width=1.0 * pq.s):
        Kernel.__init__(self, self.evaluate, half_width=half_width)

    def boundary_enclosing_at_least(self, fraction):
        return self.params['half_width']


def bin_spike_train(train, sampling_rate=None, t_start=None, t_stop=None):
    """ Creates a binned representation of a spike train.

    :param SpikeTrain train: Spike train to bin.
    :param sampling_rate: The sampling rate which will be used to bin
        the spike train. If `None`, `train.sampling_rate` will be used. If that
        is also `None`, :py:const:`default_sampling_rate` will be used.
    :type sampling_rate: Quantity scalar
    :param t_start: Time point of the left boundary of the first bin. If `None`,
        `train.t_start` will be used.
    :type t_start: Quantity scalar
    :param t_stop: Timp point of the right boundary of the last bin. If `None`,
        `train.t_stop` will be used.
    :type t_stop: Quantity scalar
    :returns: The binned representation of the spike train, the boundaries of
        the discretization bins
    :rtype: (1D array, Quantity 1D)
    """

    if sampling_rate is None:
        if train.sampling_rate is not None:
            sampling_rate = train.sampling_rate
        else:
            sampling_rate = default_sampling_rate
    if t_start is None:
        t_start = train.t_start
    if t_stop is None:
        t_stop = train.t_stop

    duration = t_stop - t_start
    num_bins = sampling_rate * duration + 1
    bins = sp.linspace(t_start, t_stop, num_bins)
    binned, _ = sp.histogram(train.rescale(bins.units), bins)
    return binned, bins


def st_convolve(
        train, kernel, kernel_area_fraction=default_kernel_area_fraction,
        mode='same', **discretizationParams):
    """ Convolves a spike train with a kernel.

    :param SpikeTrain train: Spike train to convolve.
    :param kernel: The kernel instance to convolve with.
    :type kernel: :class:`Kernel`
    :param float kernel_area_fraction: A value between 0 and 1 which controls
        the interval over which the kernel will be discretized. At least the
        given fraction of the complete kernel area will be covered. Higher
        values can lead to more accurate results (besides the sampling rate).
    :param mode:
        * 'same': The default which returns an array covering the whole
          the duration of the spike train `train`.
        * 'full': Returns an array with additional discretization bins in the
          beginning and end so that for each spike the whole discretized
          kernel is included.
        * 'valid': Returns only the discretization bins where the discretized
          kernel and spike train completely overlap.

        See also `numpy.convolve
        <http://docs.scipy.org/doc/numpy/reference/generated/numpy.convolve.html>`_.
    :type mode: {'same', 'full', 'valid'}
    :param discretizationParams: Additional discretization arguments which will
        be passed to :func:`bin_spike_train`.
    :returns: The convolved spike train, the boundaries of the discretization
        bins
    :rtype: (Quantity 1D, Quantity 1D)
    """

    binned, bins = bin_spike_train(train, **discretizationParams)
    sampling_rate = binned.size / (bins[-1] - bins[0])
    k = kernel.discretize(kernel_area_fraction, sampling_rate)
    result = scipy.signal.convolve(binned, k, mode) * k.units

    assert (result.size - binned.size) % 2 == 0
    num_additional_bins = (result.size - binned.size) // 2
    bins = sp.linspace(
        bins[0] - num_additional_bins / sampling_rate,
        bins[-1] + num_additional_bins / sampling_rate,
        result.size + 1)

    return result, bins
