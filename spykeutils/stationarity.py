"""
.. autofunction:: spike_amplitude_histogram(trains, num_bins, uniform_y_scale=True, unit=uV, progress=None)
"""
import scipy as sp
import quantities as pq

from progress_indicator import ProgressIndicator
from . import SpykeException


def spike_amplitude_histogram(trains, num_bins, uniform_y_scale=True,
                              unit=pq.uV, progress=None):
    """ Return a spike amplitude histogram.

    The resulting is useful to assess the drift in spike amplitude over a longer
    recording. It shows histograms (one for each ``trains`` entry, e.g. segment)
    of maximum and minimum spike amplitudes.

    :param list trains: A list of lists of SpikeTrain objects. Each entry of
        the outer list will be one point on the x-axis (they could correspond
        to segments), all amplitude occurences of spikes contained in the
        inner list will be added up.
    :param int num_bins: Number of bins for the histograms.
    :param bool uniform_y_scale: If True, the histogram for each channel
        will use the same bins. Otherwise, the minimum bin range is computed
        separately for each channel.
    :param Quantity unit: Unit of Y-Axis.
    :param progress: Set this parameter to report progress.
    :type progress: :class:`spykeutils.progress_indicator.ProgressIndicator`
    :return: A tuple with three values:

        * A three-dimensional histogram matrix, where the first dimension
          corresponds to bins, the second dimension to the entries of
          ``trains`` (e.g. segments) and the third dimension to channels.
        * A list of the minimum amplitude value for each channel (all values
          will be equal if ``uniform_y_scale`` is true).
        * A list of the maximum amplitude value for each channel (all values
          will be equal if ``uniform_y_scale`` is true).
    :rtype: (ndarray, list, list)
    """
    if not progress:
        progress = ProgressIndicator()

    num_channels = 1
    for t in trains:
        if not t:
            continue
        num_channels = t[0].waveforms.shape[2]
        break

    progress.set_ticks(2*len(trains))
    progress.set_status('Calculating Spike Amplitude Histogram')

    # Find maximum and minimum amplitudes on all channels
    up = [0] * num_channels
    down = [0] * num_channels
    for t in trains:
        for s in t:
            if s.waveforms is None:
                continue
            if s.waveforms.shape[2] != num_channels:
                raise SpykeException('All spikes need to have the same ' +
                                     'numer of channels for Spike Amplitude Histogram!')
            a = sp.asarray(s.waveforms.rescale(unit))
            u = a.max(1)
            d = a.min(1)
            for c in xrange(num_channels):
                up[c] = max(up[c], sp.stats.mstats.mquantiles(
                    u[:,c], [0.999])[0])
                down[c] = min(down[c], sp.stats.mstats.mquantiles(
                    d[:,c], [0.001])[0])
            progress.step()

    if uniform_y_scale:
        up = [max(up)] * num_channels
        down = [min(down)] * num_channels

    # Create histogram
    bins = [sp.linspace(down[c],up[c], num_bins+1)
            for c in xrange(num_channels)]
    hist = sp.zeros((num_bins, len(trains), num_channels))
    for i, t in enumerate(trains):
        for s in t:
            if s.waveforms is None:
                continue
            a = sp.asarray(s.waveforms.rescale(unit))
            upper = a.max(1)
            lower = a.min(1)
            for c in xrange(num_channels):
                hist[:,i,c] += sp.histogram(upper[:,c], bins[c])[0]
                hist[:,i,c] += sp.histogram(lower[:,c], bins[c])[0]
        progress.step()

    return hist, down, up
