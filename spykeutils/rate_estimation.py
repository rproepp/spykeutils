"""
.. autofunction:: binned_spike_trains(trains, bin_size, start=0 ms, stop=None)
.. autofunction:: psth(trains, bin_size, rate_correction=True, start=0 ms, stop=None)
.. autofunction:: spike_density_estimation(trains, start=0 ms, stop=None, evaluation_points=None, kernel=gauss_kernel, kernel_size=100 ms, optimize_steps=None, progress=None)
"""
from __future__ import division

import scipy as sp
import quantities as pq
import neo
from progress_indicator import ProgressIndicator
from . import SpykeException


def _binned_spike_trains(trains, bins):
    """ Return a binned representation of spike trains.

    :param sequencs trains: A sequence of :class:`neo.core.SpikeTrain`
        objects.
    :param bins: The bin edges, including the rightmost edge.
    :type bins: Quantity 1D
    :returns: List of one-dimensional arrays of spike counts.
    :rtype: list
    """
    counts = []
    for t in trains:
        counts.append(sp.histogram(t.rescale(bins.units), bins)[0])

    return counts


def binned_spike_trains(trains, bin_size, start=0*pq.ms, stop=None):
    """ Return dictionary of binned rates for a dictionary of
    spike train lists.

    :param dict trains: A dictionary of :class:`neo.core.SpikeTrain` lists.
    :param bin_size: The desired bin size (as a time quantity).
    :type bin_size: Quantity scalar
    :type start: The desired time for the start of the first bin.
        It will be recalculated if there are spike trains which
        start later than this time.
    :type start: Quantity scalar
    :param stop: The desired time for the end of the last bin. It will
        be recalculated if there are spike trains which end earlier
        than this time.
    :type stop: Quantity scalar
    :returns: A dictionary (with the same indices as ``trains``) of lists
        of spike train counts and the bin borders.
    :rtype: dict, Quantity 1D
    """
    # Do not create bins that do not include all spike trains
    max_start, max_stop = minimum_spike_train_interval(trains)

    start = max(start, max_start)
    start.units = bin_size.units
    if stop is not None:
        stop = min(stop, max_stop)
    else:
        stop = max_stop
    stop.units = bin_size.units

    # Calculate bin size
    bins = sp.arange(start, stop, bin_size)*bin_size.units

    # Create dictionary for all SpikeTrain lists
    binned = {}
    for s in trains:
        b = _binned_spike_trains(trains[s], bins)
        if b:
            binned[s] = b

    return binned, bins


def psth(trains, bin_size, rate_correction=True, start=0*pq.ms, stop=None):
    """ Return dictionary of peri stimulus time histograms for a dictionary
    of spike train lists.

    :param dict trains: A dictionary of lists of :class:`neo.core.SpikeTrain`
        objects.
    :param bin_size: The desired bin size (as a time quantity).
    :type bin_size: Quantity scalar
    :param bool rate_correction: Determines if a rates (``True``) or
        counts (``False``) are returned.
    :param start: The desired time for the start of the first bin. It
        will be recalculated if there are spike trains which start
        later than this time.
    :type start: Quantity scalar
    :param stop: The desired time for the end of the last bin. It will
        be recalculated if there are spike trains which end earlier
        than this time.
    :type stop: Quantity scalar
    :returns: A dictionary (with the same indices as ``trains``) of arrays
        containing counts (or rates if ``rate_correction`` is ``True``)
        and the bin borders.
    :rtype: dict, Quantity 1D
    """
    if not trains:
        raise SpykeException('No spike trains for PSTH!')

    binned, bins = binned_spike_trains(trains, bin_size, start, stop)

    cumulative = {}
    time_multiplier = 1.0 / float(bin_size.rescale(pq.s))
    for u in binned:
        if rate_correction:
            cumulative[u] = sp.mean(sp.array(binned[u]), 0)
        else:
            cumulative[u] = sp.sum(sp.array(binned[u]), 0)
        cumulative[u] *= time_multiplier

    return cumulative, bins


def aligned_spike_trains(trains, events, copy=True):
    """ Return a list of spike trains aligned to an event (the event will
    be time 0 on the returned trains).

    :param dict trains: A list of :class:`neo.core.SpikeTrain` objects.
    :param dict events: A dictionary of Event objects, indexed by segment.
        These events (in case of lists, always the first element in the list)
        will be used to align the spike trains and will be at time 0 for
        the aligned spike trains.
    :param bool copy: Determines if aligned copies of the original
        spike trains  will be returned. If not, every spike train needs
        exactly one corresponding event, otherwise a ``ValueError`` will
        be raised. Otherwise, entries with no event will be ignored.
    """
    ret = []
    for t in trains:
        s = t.segment
        if s not in events:
            if not copy:
                raise ValueError(
                    'Cannot align spike trains: At least one segment does' +
                    'not have an align event.')
            continue

        if copy:
            st = t.rescale(t.units)
        else:
            st = t

        e = events[s]
        st -= e.time
        st.t_stop -= e.time
        st.t_start -= e.time
        ret.append(st)

    return ret


def minimum_spike_train_interval(trains):
    """ Computes the minimum starting time and maximum end time that all
    given spike trains share.

    :param dict trains: A dictionary of sequences of
        :class:`neo.core.SpikeTrain` objects.
    :returns: Maximum shared start time and minimum shared stop time.
    :rtype: Quantity scalar, Quantity scalar
    """
    start = -sp.inf * pq.s
    stop = sp.inf * pq.s

    # Load data and find shortest spike train
    for st in trains.itervalues():
        # Minimum length of spike of all spike trains for this unit
        start = max(start, max((t.t_start for t in st)))
        stop = min(stop, min((t.t_stop for t in st)))

    return start, stop


def gauss_kernel(x, kernel_size):
    return 1.0 / (sp.sqrt(2*sp.pi) * kernel_size) * \
           sp.exp(-0.5 *  (x / kernel_size)**2)


def _hist_density(hist, kernel, ksize, start, stop):
    time_factor = (2048.0 * start.units / (stop - start)).simplified
    kern = kernel(sp.arange(-1024,1024), ksize * time_factor)
    kern *= time_factor / sp.sum(kern)
    kde = sp.signal.convolve(hist, kern, 'same')
    return kde


def _train_density(train, kernel, ksize):
    bins = sp.linspace(train.t_start, train.t_stop, 1025)
    hist = sp.histogram(train, bins)[0]
    return _hist_density(hist, kernel, ksize, train.t_start, train.t_stop)


def spike_density_estimation(trains, start=0*pq.ms, stop=None,
                             kernel=gauss_kernel, kernel_size=100*pq.ms,
                             optimize_steps=None, progress=None):
    """ Create a spike density estimation from a dictionary of
    lists of spike trains.

    The spike density estimations give an estimate of the instantaneous
    rate. The density estimation is evaluated at 1024 equally spaced
    points covering the range of the input spike trains. Optionally finds
    optimal kernel size for given data using the algorithm from
    (Shimazaki, Shinomoto. Journal of Computational Neuroscience. 2010).

    :param dict trains: A dictionary of :class:`neo.core.SpikeTrain` lists.
    :param start: The desired time for the start of the estimation. It
        will be recalculated if there are spike trains which start later
        than this time. This parameter can be negative (which could be
        useful when aligning on events).
    :type start: Quantity scalar
    :param stop: The desired time for the end of the estimation. It will
        be recalculated if there are spike trains which end earlier
        than this time.
    :type stop: Quantity scalar
    :param func kernel: The kernel function to use, should accept
        two parameters: A ndarray of distances and a kernel size.
        The total area under the kernel function sould be 1.
        Default: Gaussian kernel
    :param kernel_size: A uniform kernel size for all spike trains.
            Only used if optimization of kernel sizes is not used.
    :type kernel_size: Quantity scalar
    :param optimize_steps: An array of time lengths that will be
        considered in the kernel width optimization. Note that the
        optimization assumes a Gaussian kernel and will most likely
        not give the optimal kernel size if another kernel is used.
        If None, ``kernel_size`` will be used.
    :type optimize_steps: Quantity 1D
    :param progress: Set this parameter to report progress.
    :type progress: :class:`spykeutils.progress_indicator.ProgressIndicator`

    :returns: Three values:

        * A dictionary of the spike density estimations (Quantity 1D in
          Hz). Indexed the same as ``trains``.
        * A dictionary of kernel sizes (Quantity scalars). Indexed the
          same as ``trains``.
        * The used evaluation points.
    :rtype: dict, dict, Quantity 1D
    """
    if not progress:
        progress = ProgressIndicator()

    if optimize_steps is None or len(optimize_steps) < 1:
        units = kernel_size.units
    else:
        units = optimize_steps.units

    # Prepare evaluation points
    max_start, max_stop = minimum_spike_train_interval(trains)

    start = max(start, max_start)
    start.units = units
    if stop is not None:
        stop = min(stop, max_stop)
    else:
        stop = max_stop
    stop.units = units
    bins = sp.linspace(start, stop, 1025)
    eval_points = bins[:-1] + (bins[1] - bins[0]) / 2

    if optimize_steps is None or len(optimize_steps) < 1:
        kernel_size = {u:kernel_size for u in trains}
    else:
        # Find optimal kernel size for all spike train sets
        progress.set_ticks(len(optimize_steps)*len(trains))
        progress.set_status('Calculating optimal kernel size')
        kernel_size = {}
        for u,t in trains.iteritems():
            c = collapsed_spike_trains(t)
            kernel_size[u] = optimal_gauss_kernel_size(
                c.time_slice(start,stop), optimize_steps, progress)

    progress.set_ticks(len(trains))
    progress.set_status('Creating spike density plot')

    # Calculate KDEs
    kde = {}
    for u,t in trains.iteritems():
        # Collapse spike trains
        collapsed = collapsed_spike_trains(t).rescale(units)
        ksize = float(kernel_size[u])

        # Create density estimation using convolution
        kde[u] = _train_density(collapsed.time_slice(start, stop),
            kernel, ksize) / len(trains[u]) / units
        kde[u].units = pq.Hz
    return kde, kernel_size, eval_points


def collapsed_spike_trains(trains):
    """ Return a superposition of a list of spike trains.

    :param iterable trains: A list of :class:`neo.core.SpikeTrain` objects
    :returns: A spike train object containing all spikes of the given
        spike trains.
    :rtype: :class:`neo.core.SpikeTrain`
    """
    if not trains:
        return neo.SpikeTrain([], 0)

    start = min((t.t_start for t in trains))
    stop = max((t.t_stop for t in trains))

    collapsed = []
    for t in trains:
        collapsed.extend(sp.asarray(t.rescale(stop.units)))

    return neo.SpikeTrain(collapsed*stop.units, t_stop=stop, t_start=start)


def optimal_gauss_kernel_size(train, optimize_steps, progress=None):
    """ Return the optimal kernel size for a spike density estimation
    of a spike train for a gaussian kernel. This function takes a single
    spike train, which can be a superposition of multiple spike trains
    (created with :func:`collapsed_spike_trains`) that should be included
    in a spike density estimation.

    Implements the algorithm from
    (Shimazaki, Shinomoto. Journal of Computational Neuroscience. 2010).

    :param train: The spike train for which the kernel
        size should be optimized.
    :type train: :class:`neo.core.SpikeTrain`
    :param optimize_steps: Array of kernel sizes to try (the best of
        these sizes will be returned).
    :type optimize_steps: Quantity 1D
    :param progress: Set this parameter to report progress. Will be
        advanced by len(`optimize_steps`) steps.
    :type progress: :class:`spykeutils.progress_indicator.ProgressIndicator`
    :returns: Best of the given kernel sizes
    :rtype: Quantity scalar
    """
    if not progress:
        progress = ProgressIndicator()

    x = train.rescale(optimize_steps.units)
    steps = sp.asarray(optimize_steps)

    N = len(train)
    C = {}

    bins = sp.linspace(x.t_start, x.t_stop, 1025)
    dt = float(bins[1] - bins[0])
    y_hist = sp.histogram(x, bins)[0] / N / dt
    for step in steps:
        s = float(step)
        yh = _hist_density(y_hist, gauss_kernel, step,
            train.t_start, train.t_stop)

        # Equation from Matlab code, 7/2012
        c = (sp.sum(yh**2) * dt -
             2 * sp.sum(yh * y_hist) * dt +
             2 * 1 / sp.sqrt(2 * sp.pi) / s / N)
        C[s] = c * N * N
        progress.step()

    # Return kernel size with smallest cost
    return min(C, key=C.get)*train.units
