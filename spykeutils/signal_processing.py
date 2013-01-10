
import quantities as pq
import scipy as sp
import scipy.signal


def causal_decaying_exp_kernel(x, kernel_size=1.0 * pq.s):
    return sp.piecewise(
        x, [x < 0, x >= 0], [
            lambda x: 0,
            lambda x: sp.exp(-x / kernel_size) / kernel_size])


def gauss_kernel(x, kernel_size=1.0 * pq.s):
    return (1.0 / (sp.sqrt(2 * sp.pi) * kernel_size) *
            sp.exp(-1 / 2 * (x / kernel_size) ** 2))


def laplace_kernel(x, kernel_size=1.0 * pq.s):
    return sp.exp(-sp.absolute(x) / kernel_size) / (2.0 * kernel_size)


def rectangular_kernel(x, half_width=1.0 * pq.s):
    return (sp.absolute(x) < half_width) / (2.0 * half_width)


class Kernel(object):
    def __init__(self, kernel_func, **params):
        self.kernel_func = kernel_func
        self.params = params

    def __call__(self, x):
        return self.kernel_func(x, **self.params)


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
        is also `None`, 100Hz will be used.
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
            sampling_rate = 100 * pq.Hz

    duration = train.t_stop - train.t_start
    num_bins = sampling_rate * duration + 1
    t_step = duration / num_bins
    bins = sp.linspace(train.t_start, train.t_stop, num_bins)
    binned, _ = sp.histogram(train, bins)
    k = kernel(_pq_arange(-duration, duration, t_step))
    k /= (t_step * sp.sum(k))
    return scipy.signal.convolve(binned, k, 'same'), bins
