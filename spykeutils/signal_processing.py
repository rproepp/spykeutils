"""
.. data:: default_sampling_rate
"""

import copy
import quantities as pq
import scipy as sp
import scipy.signal
import scipy.special

default_kernel_area_fraction = 0.99999
default_sampling_rate = 100 * pq.Hz


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

    def __init__(self, kernel_size, normalize):
        """
        :param kernel_size: Parameter controlling the kernel size.
        :type kernel_size: Quantity 1D
        :param bool normalize: Whether to normalize the kernel to unit area.
        """
        self.kernel_size = kernel_size
        self.normalize = normalize

    def __call__(self, t, kernel_size=None):
        """ Evaluates the kernel at all time points in the array `t`.

        :param t: Time points to evaluate the kernel at.
        :type t: Quantity 1D
        :param kernel_size: If not `None` this overwrites the kernel size of
            the `Kernel` instance.
        :type kernel_size: Quantity scalar
        :returns: The result of the kernel evaluations.
        :rtype: Quantity 1D
        """

        if kernel_size is None:
            kernel_size = self.kernel_size

        if self.normalize:
            normalization = self.normalization_factor(kernel_size)
        else:
            normalization = 1.0 * pq.dimensionless

        return self._evaluate(t, kernel_size) * normalization

    def _evaluate(self, t, kernel_size):
        """ Evaluates the kernel.

        :param t: Time points to evaluate the kernel at.
        :type t: Quantity 1D
        :param kernel_size: Controls the width of the kernel.
        :type kernel_size: Quantity scalar
        :returns: The result of the kernel evaluations.
        :rtype: Quantity 1D
        """
        raise NotImplementedError()

    def normalization_factor(self, kernel_size):
        """ Returns the factor needed to normalize the kernel to unit area.

        :param kernel_size: Controls the width of the kernel.
        :type kernel_size: Quantity scalar
        :returns: Factor to normalize the kernel to unit width.
        :rtype: Quantity scalar
        """
        raise NotImplementedError()

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

    def discretize(
            self, area_fraction=default_kernel_area_fraction,
            sampling_rate=default_sampling_rate, num_bins=None,
            ensure_unit_area=False):
        """ Discretizes the kernel.

        :param float area_fraction: Fraction between 0 and 1 (exclusive)
            of the integral of the kernel which will be at least covered by the
            discretization. Will be ignored if `num_bins` is not `None`.
        :param sampling_rate: Sampling rate for the discretization.
        :type sampling_rate: Quantity scalar
        :param int num_bins: Number of bins to use for the discretization.
        :param bool ensure_unit_area: If `True`, the area of the discretized
            kernel will be normalized to 1.0.
        :rtype: Quantity 1D
        """

        t_step = 1.0 / sampling_rate

        if num_bins is not None:
            start = -num_bins // 2
            stop = num_bins // 2
        elif area_fraction is not None:
            boundary = self.boundary_enclosing_at_least(area_fraction)
            start = sp.ceil(-boundary / t_step)
            stop = sp.floor(boundary / t_step) + 1
        else:
            raise ValueError(
                "One of area_fraction and num_bins must not be None.")

        k = self(sp.arange(start, stop) * t_step)
        if ensure_unit_area:
            k /= sp.sum(k) * t_step
        return k

    def is_symmetric(self):
        """ Should return `True` if the kernel is symmetric. """
        return False

    def summed_dist_matrix(self, vectors, presorted=False):
        """ Calculates the sum of all element pair distances for each
        pair of vectors.

        If :math:`(a_1, \\dots, a_n)` and :math:`(b_1, \\dots, b_m)` is a pair
        of vectors from `vectors` and :math:`K` the kernel, the resulting entry
        in the 2D array will be :math:`D_{ij} = \\sum_{i=1}^{n} \\sum_{j=1}^{m}
        K(a_i - b_j)`.

        :param sequence vectors: A sequence of 1D arrays to calculate the summed
            distances for each pair.
        :param bool presorted: Some optimized specializations of this function
            may need sorted vectors. Set `presorted` to `True` if you know that
            the passed vectors are already sorted to skip the sorting and thus
            increase performance.
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


class KernelFromFunction(Kernel):
    """ Creates a kernel form a function. Please note, that not all function for
    such a kernel are implemented.
    """

    def __init__(self, kernel_func, kernel_size):
        Kernel.__init__(self, kernel_size, normalize=False)
        self._evaluate = kernel_func

    def is_symmetric(self):
        return False


def as_kernel_of_size(obj, kernel_size):
    """ Returns a kernel of desired size.

    :param obj: Either an existing kernel or a kernel function. A kernel
        function takes two arguments. First a `Quantity 1D` of evaluation time
        points and second a kernel size.
    :type obj: Kernel or func
    :param kernel_size: Desired size of the kernel.
    :type kernel_size: Quantity 1D
    :returns: A :class:`Kernel` with the desired kernel size. If `obj` is
        already a :class:`Kernel` instance, a shallow copy of this instance with
        changed kernel size will be returned. If `obj` is a function it will be
        wrapped in a :class:`Kernel` instance.
    :rtype: :class:`Kernel`
    """

    if isinstance(obj, Kernel):
        obj = copy.copy(obj)
        obj.kernel_size = kernel_size
    else:
        obj = KernelFromFunction(obj, kernel_size)
    return obj


class SymmetricKernel(Kernel):
    """ Base class for kernels. """

    def __init__(self, kernel_size, normalize):
        """
        :param function kernel_func: Kernel function which takes an array
            of time points as first argument.
        :param kernel_size: Parameter controlling the kernel size.
        :param dict params: Additional parameters to be passed to the kernel
            function.
        """
        Kernel.__init__(self, kernel_size, normalize)

    def is_symmetric(self):
        return True

    def summed_dist_matrix(self, vectors, presorted=False):
        D = sp.empty((len(vectors), len(vectors)))
        if len(vectors) > 0:
            might_have_units = self(vectors[0])
            if hasattr(might_have_units, 'units'):
                D = D * might_have_units.units

        for i in xrange(len(vectors)):
            for j in xrange(i, len(vectors)):
                D[i, j] = D[j, i] = sp.sum(self(
                    (vectors[i] - sp.atleast_2d(vectors[j]).T).flatten()))
        return D


class CausalDecayingExpKernel(Kernel):
    @staticmethod
    def evaluate(t, kernel_size):
        return sp.piecewise(
            t, [t < 0, t >= 0], [
                lambda t: 0,
                lambda t: sp.exp(
                    (-t * pq.dimensionless / kernel_size).simplified)])

    def _evaluate(self, t, kernel_size):
        return self.evaluate(t, kernel_size)

    def normalization_factor(self, kernel_size):
        return 1.0 / kernel_size

    def __init__(self, kernel_size=1.0 * pq.s, normalize=True):
        Kernel.__init__(self, kernel_size, normalize)

    def boundary_enclosing_at_least(self, fraction):
        return -self.kernel_size * sp.log(1.0 - fraction)


class GaussianKernel(SymmetricKernel):
    @staticmethod
    def evaluate(t, kernel_size):
        return sp.exp(
            -0.5 * (t * pq.dimensionless / kernel_size).simplified ** 2)

    def _evaluate(self, t, kernel_size):
        return self.evaluate(t, kernel_size)

    def normalization_factor(self, kernel_size):
        return 1.0 / (sp.sqrt(2.0 * sp.pi) * kernel_size)

    def __init__(self, kernel_size=1.0 * pq.s, normalize=True):
        Kernel.__init__(self, kernel_size, normalize)

    def boundary_enclosing_at_least(self, fraction):
        return self.kernel_size * sp.sqrt(2.0) * \
            scipy.special.erfinv(fraction + scipy.special.erf(0.0))


class LaplacianKernel(SymmetricKernel):
    @staticmethod
    def evaluate(t, kernel_size):
        return sp.exp(
            -(sp.absolute(t) * pq.dimensionless / kernel_size).simplified)

    def _evaluate(self, t, kernel_size):
        return self.evaluate(t, kernel_size)

    def normalization_factor(self, kernel_size):
        return 0.5 / kernel_size

    def __init__(self, kernel_size=1.0 * pq.s, normalize=True):
        Kernel.__init__(self, kernel_size, normalize)

    def boundary_enclosing_at_least(self, fraction):
        return -self.kernel_size * sp.log(1.0 - fraction)

    def summed_dist_matrix(self, vectors, presorted=False):
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

        if not presorted:
            vectors = [v.copy() for v in vectors]
            for v in vectors:
                v.sort()

        exp_vecs = [sp.exp(v / self.kernel_size) for v in vectors]
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

        if self.normalize:
            normalization = self.normalization_factor(self.kernel_size)
        else:
            normalization = 1.0
        return normalization * D


class RectangularKernel(SymmetricKernel):
    @staticmethod
    def evaluate(t, half_width):
        return (sp.absolute(t) < half_width)

    def _evaluate(self, t, kernel_size):
        return self.evaluate(t, kernel_size)

    def normalization_factor(self, half_width):
        return 0.5 / half_width

    def __init__(self, half_width=1.0 * pq.s, normalize=True):
        Kernel.__init__(self, half_width, normalize)

    def boundary_enclosing_at_least(self, fraction):
        return self.kernel_size


class TriangularKernel(SymmetricKernel):
    @staticmethod
    def evaluate(t, half_width):
        return sp.maximum(
            0.0,
            (1.0 - sp.absolute(t) * pq.dimensionless / half_width).magnitude)

    def _evaluate(self, t, kernel_size):
        return self.evaluate(t, kernel_size)

    def normalization_factor(self, half_width):
        return 1.0 / half_width

    def __init__(self, half_width=1.0 * pq.s, normalize=True):
        Kernel.__init__(self, half_width, normalize)

    def boundary_enclosing_at_least(self, fraction):
        return self.kernel_size


def bin_spike_train(
        train, sampling_rate=default_sampling_rate, t_start=None, t_stop=None):
    """ Creates a binned representation of a spike train.

    :param SpikeTrain train: Spike train to bin.
    :param sampling_rate: The sampling rate which will be used to bin
        the spike train.
    :type sampling_rate: Quantity scalar
    :param t_start: Time point of the left boundary of the first bin. If `None`,
        `train.t_start` will be used.
    :type t_start: Quantity scalar
    :param t_stop: Time point of the right boundary of the last bin. If `None`,
        `train.t_stop` will be used.
    :type t_stop: Quantity scalar
    :returns: The binned representation of the spike train, the boundaries of
        the discretization bins
    :rtype: (1D array, Quantity 1D)
    """

    if t_start is None:
        t_start = train.t_start
    if t_stop is None:
        t_stop = train.t_stop

    duration = t_stop - t_start
    num_bins = sampling_rate * duration + 1
    bins = sp.linspace(t_start, t_stop, num_bins)
    binned, dummy = sp.histogram(train.rescale(bins.units), bins)
    return binned, bins


def smooth(
        binned, kernel, sampling_rate, mode='same',
        **kernel_discretization_params):
    """ Smoothes a binned representation (e.g. of a spike train) by convolving
    with a kernel.

    :param binned: Bin array to smooth.
    :type binned: 1-D array
    :param kernel: The kernel instance to convolve with.
    :type kernel: :class:`Kernel`
    :param sampling_rate: The sampling rate which will be used to discretize the
        kernel. It should be equal to the sampling rate used to obtain `binned`.
    :type sampling_rate: Quantity scalar
    :param mode:
        * 'same': The default which returns an array of the same size as
          `binned`
        * 'full': Returns an array with a bin for each shift where `binned` and
          the discretized kernel overlap by at least one bin.
        * 'valid': Returns only the discretization bins where the discretized
          kernel and `binned` completely overlap.

        See also `numpy.convolve
        <http://docs.scipy.org/doc/numpy/reference/generated/numpy.convolve.html>`_.
    :type mode: {'same', 'full', 'valid'}
    :param dict kernel_discretization_params: Additional discretization
        arguments which will be passed to :meth:`.Kernel.discretize`.
    :returns: The smoothed representation of `binned`.
    :rtype: Quantity 1D
    """
    k = kernel.discretize(
        sampling_rate=sampling_rate, **kernel_discretization_params)
    return scipy.signal.convolve(binned, k, mode) * k.units


def st_convolve(
        train, kernel, sampling_rate=default_sampling_rate,
        mode='same', binning_params={}, kernel_discretization_params={}):
    """ Convolves a spike train with a kernel.

    :param SpikeTrain train: Spike train to convolve.
    :param kernel: The kernel instance to convolve with.
    :type kernel: :class:`Kernel`
    :param sampling_rate: The sampling rate which will be used to bin
        the spike train.
    :type sampling_rate: Quantity scalar
    :param mode:
        * 'same': The default which returns an array covering the whole
          duration of the spike train `train`.
        * 'full': Returns an array with additional discretization bins in the
          beginning and end so that for each spike the whole discretized
          kernel is included.
        * 'valid': Returns only the discretization bins where the discretized
          kernel and spike train completely overlap.

        See also `numpy.convolve
        <http://docs.scipy.org/doc/numpy/reference/generated/numpy.convolve.html>`_.
    :type mode: {'same', 'full', 'valid'}
    :param dict binning_params: Additional discretization arguments which will
        be passed to :func:`bin_spike_train`.
    :param dict kernel_discretization_params: Additional discretization
        arguments which will be passed to :meth:`.Kernel.discretize`.
    :returns: The convolved spike train, the boundaries of the discretization
        bins
    :rtype: (Quantity 1D, Quantity 1D)
    """

    binned, bins = bin_spike_train(train, sampling_rate, **binning_params)
    sampling_rate = binned.size / (bins[-1] - bins[0])
    result = smooth(
        binned, kernel, sampling_rate, mode, **kernel_discretization_params)

    assert (result.size - binned.size) % 2 == 0
    num_additional_bins = (result.size - binned.size) // 2
    bins = sp.linspace(
        bins[0] - num_additional_bins / sampling_rate,
        bins[-1] + num_additional_bins / sampling_rate,
        result.size + 1)

    return result, bins
