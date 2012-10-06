"""
.. autofunction:: correlogram(trains, bin_size, max_lag=500 ms, border_correction=True, unit=ms, progress=None)
"""
import scipy as sp
from collections import OrderedDict

import quantities as pq

from progress_indicator import ProgressIndicator
from . import SpykeException

def correlogram(trains, bin_size, max_lag=500*pq.ms, border_correction=True,
                unit=pq.ms, progress=None):
    """ Return (cross-)correlograms from a dictionary of SpikeTrain
    lists for different units.

    :param dict trains: Dictionary of SpikeTrain lists.
    :param bin_size: Bin size (time).
    :type bin_size: Quantity scalar
    :param max_lag: Cut off (end time of calculated correlogram).
    :type max_lag: Quantity scalar
    :param bool border_correction: Apply correction for less data at higher
        timelags. Not perfect for bin_size != 1*``unit``, especially with
        large ``max_lag`` compared to length of spike trains.
    :param Quantity unit: Unit of X-Axis.
    :param progress: A ProgressIndicator object for the operation.
    :type progress: :class:`spykeutils.progress_indicator.ProgressIndicator`
    :returns: Two values:

        * An ordered dictionary indexed with the indices of ``trains`` of
          ordered dictionaries indexed with the same indices. Entries of
          the inner dictionaries are the resulting (cross-)correlograms as
          numpy arrays. All crosscorrelograms can be indexed in two
          different ways: ``c[index1][index2]`` and ``c[index2][index1]``.
        * The bins used for the correlogram calculation.
    :rtype: dict, Quantity 1D
    """
    if not progress:
        progress = ProgressIndicator()

    bin_size.rescale(unit)
    max_lag.rescale(unit)

    # Create bins, making sure that 0 is at the center of central bin
    half_bins = sp.arange(bin_size / 2, max_lag, bin_size)
    all_bins = list(reversed(-half_bins))
    all_bins.extend(half_bins)
    bins = sp.array(all_bins) * unit
    middle_bin = len(bins) / 2 - 1

    indices = sorted(trains.keys(), key=lambda (u):u.name if u else None)
    num_trains = len(trains[indices[0]])
    if not num_trains:
        raise SpykeException('Could not create correlogram: No spike trains!')
    for u in range(1, len(indices)):
        if len(trains[indices[u]]) != num_trains:
            raise SpykeException('Could not create correlogram: All units ' +
                                 'need the same number of spike trains!')

    progress.set_ticks(sp.sum(range(len(trains) + 1) * num_trains))

    corrector = 1
    if border_correction:
        # Need safe min/max functions
        def safe_max(seq):
            if len(seq) < 1:
                return 0
            return max(seq)
        def safe_min(seq):
            if len(seq) < 1:
                return 2**20 #Some arbitrary large value
            return min(seq)

        max_w = max([max([safe_max(t) for t in l])
                     for l in trains.itervalues()])
        min_w = min([min([safe_min(t) for t in l])
                     for l in trains.itervalues()])

        train_length = (max_w - min_w)
        l = int(round(middle_bin)) + 1
        cE = max(train_length-(l*bin_size)+1*unit, 1*unit)

        corrector = train_length / sp.concatenate(
            (sp.linspace(cE, train_length, l-1, False),
             sp.linspace(train_length, cE, l)))

    correlograms = OrderedDict()
    for i1 in xrange(len(indices)): # For each index
        # For all later indices, including itself
        for i2 in xrange(i1, len(indices)):
            histogram = sp.zeros(len(bins) - 1)
            for t in xrange(num_trains):
                train2 = trains[indices[i2]][t].rescale(unit)
                for s in trains[indices[i1]][t]:
                    histogram += sp.histogram(train2,
                        bins + s.rescale(unit))[0]
                if i1 == i2: # Correction for autocorrelogram
                    histogram[middle_bin] -= len(train2)

                progress.step()
            crg = corrector*histogram/num_trains
            if indices[i1] not in correlograms:
                correlograms[indices[i1]] = OrderedDict()
            correlograms[indices[i1]][indices[i2]] = crg
            if i1 != i2:
                if indices[i2] not in correlograms:
                    correlograms[indices[i2]] = OrderedDict()
                correlograms[indices[i2]][indices[i1]] = crg

    return correlograms, bins