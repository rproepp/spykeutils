"""
.. data:: default_sampling_rate
"""

import quantities as pq
import scipy as sp
import scipy.signal

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

    def times_to_fall_below(self, threshold):
        """ Calculates the time shifts it which the kernel value falls below a
        certain threshold.

        :param threshold:
        :type threshold: Quanitity scalar
        :returns: left bound, right bound
        :rtype: Quantity scalar, Quantity scalar
        """
        raise NotImplementedError()


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

    def times_to_fall_below(self, threshold):
        return (0.0 * self.params['kernel_size'].units,
            -self.params['kernel_size'] * scipy.log(
                (self.params['kernel_size'] * threshold).simplified))


class GaussianKernel(Kernel):
    @staticmethod
    def evaluate(t, kernel_size):
        return (1.0 / (sp.sqrt(2 * sp.pi) * kernel_size) *
                sp.exp(-0.5 * (t / kernel_size).simplified ** 2))

    def __init__(self, kernel_size=1.0 * pq.s):
        Kernel.__init__(self, self.evaluate, kernel_size=kernel_size)

    def times_to_fall_below(self, threshold):
        t = self.params['kernel_size'] * sp.sqrt(-2 * sp.log(
            (self.params['kernel_size'] * threshold).simplified *
            sp.sqrt(2 * sp.pi)))
        return (-t, t)


class LaplacianKernel(Kernel):
    @staticmethod
    def evaluate(t, kernel_size):
        return sp.exp(-(sp.absolute(t) / kernel_size).simplified) \
                / (2.0 * kernel_size)

    def __init__(self, kernel_size=1.0 * pq.s):
        Kernel.__init__(self, self.evaluate, kernel_size=kernel_size)

    def times_to_fall_below(self, threshold):
        t = -self.params['kernel_size'] * sp.log(
            2 * (self.params['kernel_size'] * threshold).simplified)
        return (-t, t)


class RectangularKernel(Kernel):
    @staticmethod
    def evaluate(t, half_width):
        return (sp.absolute(t) < half_width) / (2.0 * half_width)

    def __init__(self, half_width=1.0 * pq.s):
        Kernel.__init__(self, self.evaluate, half_width=half_width)

    def times_to_fall_below(self, threshold):
        return (-self.params['half_width'], self.params['half_width'])


def _pq_arange(start, stop=None, step=1):
    if stop is None:
        stop = start
        start = 0 * stop.units
    return sp.arange(
        start.rescale(stop.units), stop, step.rescale(stop.units)) * stop.units


def st_convolve(train, kernel, sampling_rate=None):
    """ Convolves a spike train with a kernel.

    :param SpikeTrain train: Spike train to convolve.
    :param function kernel: The kernel function to convolve with. It has to
        accept a `Quantity 1D` as first argument giving the time points at which
        the kernel will be evaluated.
    :param sampling_rate: The sampling rate which will be used to discretize
        the spike train. If `None`, `train.sampling_rate` will be used. If that
        is also `None`, :py:const:`default_sampling_rate` will be used.
    :type sampling_rate: Quantity scalar
    :param kwargs: Additional arguments passed to the kernel function.
    :returns: The convolved spike train, the boundaries of the discretization
        bins
    :rtype: (1D array, Quantity 1D)
    """

    if sampling_rate is None:
        if train.sampling_rate is not None:
            sampling_rate = train.sampling_rate
        else:
            sampling_rate = default_sampling_rate

    duration = train.t_stop - train.t_start
    num_bins = sampling_rate * duration + 1
    t_step = duration / num_bins
    bins = sp.linspace(train.t_start, train.t_stop, num_bins)
    binned, _ = sp.histogram(train.rescale(bins.units), bins)
    k = kernel(_pq_arange(-duration, duration, t_step))
    k /= (t_step * sp.sum(k))
    return scipy.signal.convolve(binned, k, 'same'), bins
