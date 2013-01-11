"""
.. data:: default_sampling_rate
"""

import quantities as pq
import scipy as sp
import scipy.signal
import scipy.special

default_sampling_rate = 1000 * pq.Hz


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

    def interval_enclosing_at_least(self, fraction):
        """ Calculates the interval enclosing a certain fraction of the integral
        of the kernel.

        :param float fraction: Fraction of the whole area which at least has to
            be enclosed.
        :returns: left bound, right bound
        :rtype: Quantity scalar, Quantity scalar
        """
        raise NotImplementedError()

    def discretize(self, area_fraction, sampling_rate=default_sampling_rate):
        """ Discretizes the kernel.

        :param float area_fraction: Fraction between 0 and 1 (exclusive)
            of the integral of the kernel which will be at least covered by the
            discretization.
        :param sampling_rate: Sampling rate for the discretization.
        :type sampling_rate: Quantity scalar
        :rtype: 1D array
        """

        t_step = 1.0 / sampling_rate
        t_start, t_stop = self.interval_enclosing_at_least(area_fraction)
        t_start = sp.ceil(t_start / t_step) * t_step
        t_stop = sp.floor(t_stop / t_step) * t_step + t_step
        k = self(_pq_arange(t_start, t_stop, t_step))
        return k / (t_step * sp.sum(k))


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

    def interval_enclosing_at_least(self, fraction):
        return (0.0 * self.params['kernel_size'].units,
                - self.params['kernel_size'] * sp.log(1.0 - fraction))


class GaussianKernel(Kernel):
    @staticmethod
    def evaluate(t, kernel_size):
        return (1.0 / (sp.sqrt(2 * sp.pi) * kernel_size) *
                sp.exp(-0.5 * (t / kernel_size).simplified ** 2))

    def __init__(self, kernel_size=1.0 * pq.s):
        Kernel.__init__(self, self.evaluate, kernel_size=kernel_size)

    def interval_enclosing_at_least(self, fraction):
        t = self.params['kernel_size'] * sp.sqrt(2.0) * \
            scipy.special.erfinv(fraction + scipy.special.erf(0.0))
        return (-t, t)


class LaplacianKernel(Kernel):
    @staticmethod
    def evaluate(t, kernel_size):
        return sp.exp(-(sp.absolute(t) / kernel_size).simplified) \
            / (2.0 * kernel_size)

    def __init__(self, kernel_size=1.0 * pq.s):
        Kernel.__init__(self, self.evaluate, kernel_size=kernel_size)

    def interval_enclosing_at_least(self, fraction):
        t = - self.params['kernel_size'] * sp.log(1.0 - fraction)
        return (-t, t)


class RectangularKernel(Kernel):
    @staticmethod
    def evaluate(t, half_width):
        return (sp.absolute(t) < half_width) / (2.0 * half_width)

    def __init__(self, half_width=1.0 * pq.s):
        Kernel.__init__(self, self.evaluate, half_width=half_width)

    def interval_enclosing_at_least(self, fraction):
        return (-self.params['half_width'], self.params['half_width'])


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

    duration = train.t_stop - train.t_start
    num_bins = sampling_rate * duration + 1
    bins = sp.linspace(train.t_start, train.t_stop, num_bins)
    binned, _ = sp.histogram(train.rescale(bins.units), bins)
    return binned, bins


def _pq_arange(start, stop=None, step=1):
    if stop is None:
        stop = start
        start = 0 * stop.units
    return sp.arange(
        start.rescale(stop.units), stop, step.rescale(stop.units)) * stop.units


def st_convolve(
        train, kernel, kernel_area_fraction=0.999, **discretizationParams):
    """ Convolves a spike train with a kernel.

    :param SpikeTrain train: Spike train to convolve.
    :param function kernel: The kernel function to convolve with. It has to
        accept a `Quantity 1D` as first argument giving the time points at which
        the kernel will be evaluated.
    :param float kernel_area_fraction: A value between 0 and 1 which controls
        the interval over which the kernel will be discretized. At least the
        given fraction of the complete kernel area will be covered. Higher
        values can lead to more accurate results (besides the sampling rate).
    :param discretizationParams: Additional discretization arguments which will
        be passed to func:`bin_spike_train`.
    :returns: The convolved spike train, the boundaries of the discretization
        bins
    :rtype: (1D array, Quantity 1D)
    """

    binned, bins = bin_spike_train(train, **discretizationParams)
    sampling_rate = bins.size / (bins[-1] - bins[0])
    k = kernel.discretize(kernel_area_fraction, sampling_rate)
    return scipy.signal.convolve(binned, k, 'same'), bins
