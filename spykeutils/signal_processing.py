
import quantities as pq
import scipy as sp
import scipy.signal


def gauss_kernel(x, kernel_size=1.0 * pq.s):
    return (1.0 / (sp.sqrt(2 * sp.pi) * kernel_size) *
            sp.exp(-1 / 2 * (x / kernel_size) ** 2))


def rectangular_kernel(x, half_width=1.0 * pq.s):
    return (sp.absolute(x) < half_width) / (2.0 * half_width)


def pq_linspace(start, stop, num=50, endpoint=True, retstep=False):
    return sp.linspace(
        start.rescale(stop.units), stop, num, endpoint, retstep) * stop.units


def pq_arange(start, stop=None, step=1):
    if stop is None:
        stop = start
        start = 0 * stop.units
    return sp.arange(
        start.rescale(stop.units), stop, step.rescale(stop.units)) * stop.units


def st_convolve(train, kernel, sampling_rate=None, **kwargs):
    if sampling_rate is None:
        if train.sampling_rate is not None:
            sampling_rate = train.sampling_rate
        else:
            sampling_rate = 100 * pq.Hz

    duration = train.t_stop - train.t_start
    num_bins = sampling_rate * duration + 1
    t_step = duration / num_bins
    bins = pq_linspace(train.t_start, train.t_stop, num_bins)
    binned, _ = sp.histogram(train, bins)
    k = kernel(pq_arange(-duration, duration, t_step), **kwargs)
    k /= (t_step * sp.sum(k))
    return scipy.signal.convolve(binned, k, 'same')
