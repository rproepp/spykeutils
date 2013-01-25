""" Functions for estimating the quality of spike sorting results. These
functions estimate false positive and false negative fractions.
"""

from __future__ import division

import scipy as sp
from scipy.spatial.distance import cdist
import quantities as pq
import neo

from progress_indicator import ProgressIndicator
from . import SpykeException


def get_refperiod_violations(spike_trains, refperiod, progress=None):
    """ Return the refractory period violations in the given spike trains
    for the specified refractory period.

    :param dict spike_trains: Dictionary of lists of
        :class:`neo.core.SpikeTrain` objects.
    :param refperiod: The refractory period (time).
    :type refperiod: Quantity scalar
    :param progress: Set this parameter to report progress.
    :type progress: :class:`spykeutils.progress_indicator.ProgressIndicator`
    :returns: Two values:

        * The total number of violations.
        * A dictionary (with the same indices as ``spike_trains``) of
          arrays with violation times (Quantity 1D with the same unit as
          ``refperiod``) for each spike train.
    :rtype: int, dict """
    if type(refperiod) != pq.Quantity or \
       refperiod.simplified.dimensionality != pq.s.dimensionality:
        raise ValueError('refperiod must be a time quantity!')

    if not progress:
        progress = ProgressIndicator()

    total_violations = 0
    violations = {}
    for u, tL in spike_trains.iteritems():
        violations[u] = []
        for i,t in enumerate(tL):
            st = t.copy()
            st.sort()
            isi = sp.diff(st)

            violations[u].append(st[isi < refperiod].rescale(refperiod.units))
            total_violations += len(violations[u][i])

            progress.step()

    return total_violations, violations


def calculate_refperiod_fp(num_spikes, refperiod, violations, total_time):
    """ Return the rate of false positives calculated from refractory period
    calculations for each unit. The equation used is described in
    (Hill et al. The Journal of Neuroscience. 2011).

    :param dict num_spikes: Dictionary of total number of spikes,
        indexed by unit.
    :param refperiod: The refractory period (time). If the spike sorting
        algorithm includes a censored period (a time after a spike during
        which no new spikes can be found), subtract it from the refractory
        period before passing it to this function.
    :type refperiod: Quantity scalar
    :param dict violations: Dictionary of total number of violations,
        indexed the same as num_spikes.
    :param total_time: The total time in which violations could have occured.
    :type total_time: Quantity scalar

    :returns: A dictionary of false positive rates indexed by unit.
        Note that values above 0.5 can not be directly interpreted as a
        false positive rate! These very high values can e.g. indicate
        that the generating processes are not independent.
    """
    if type(refperiod) != pq.Quantity or \
       refperiod.simplified.dimensionality != pq.s.dimensionality:
        raise ValueError('refperiod must be a time quantity!')

    fp = {}
    factor = total_time / (2 * refperiod)
    for u,n in num_spikes.iteritems():
        if n == 0:
            fp[u] = 0
            continue
        zw = (violations[u] * factor / n**2).simplified

        if zw > 0.25:
            fp[u] = 0.5 + sp.sqrt(0.25 - zw).imag
            continue
        fp[u] = 0.5 - sp.sqrt(0.25 - zw)

    return fp


def _multi_norm(x, mean):
    """ Evaluate pdf of multivariate normal distribution with a mean
        at rows of x with high precision.
    """
    d = x.shape[1]
    fac = (2*sp.pi) ** (-d/2.0)
    y = cdist(x, sp.atleast_2d(mean), 'sqeuclidean') * -0.5
    return fac * sp.exp(sp.longdouble(y))


def _fast_overlap_whitened(spike_arrays, means):
    units = spike_arrays.keys()
    spikes = {u:spike_arrays[u].shape[1] for u in spike_arrays.iterkeys()}

    prior = {}
    total_spikes = 0
    for u, mean in means.iteritems():
        total_spikes += spikes[u]
    if total_spikes < 1:
        return {u: (0.0, 0.0) for u in units}, {}

    # Arrays of unnormalized posteriors (likelihood times prior)
    # for all units
    posterior = {}

    false_positive = {}
    false_negative = {}
    for u in units:
        prior[u] = spikes[u] / total_spikes
        false_positive[u] = 0
        false_negative[u] = 0

    # Calculate posteriors
    for u1 in units[:]:
        if not spikes[u1]:
            units.remove(u1)
            continue
        posterior[u1] = {}
        for u2, mean in means.iteritems():
            llh = _multi_norm(spike_arrays[u1].T, mean)
            posterior[u1][u2] = llh*prior[u2]

    # Calculate pairwise false positives/negatives
    singles = {u:{} for u in units}
    for i, u1 in enumerate(units):
        u1 = units[i]
        for u2 in units[i+1:]:
            f1 = sp.sum(posterior[u1][u2] /
                        (posterior[u1][u1] + posterior[u1][u2]),
                dtype=sp.double)

            f2 = sp.sum(posterior[u2][u1] /
                        (posterior[u2][u1] + posterior[u2][u2]),
                dtype=sp.double)

            singles[u1][u2] = (f1 / spikes[u1] if spikes[u1] else 0,
                               f2 / spikes[u1] if spikes[u1] else 0)
            singles[u2][u1] = (f2 / spikes[u2] if spikes[u2] else 0,
                               f1 / spikes[u2] if spikes[u2] else 0)

    # Calculate complete false positives/negatives with extended bayes
    for u1 in units:
        numerator = posterior[u1][u1]
        normalizer = sum(posterior[u1][u2] for u2 in units)
        false_positive[u1] = sp.sum((normalizer-numerator)/normalizer)

        other_units = units[:]
        other_units.remove(u1)
        numerator = sp.vstack((posterior[u][u1] for u in other_units))
        normalizer = sp.vstack(sum(posterior[u][u2] for u2 in units) for u in other_units)
        false_negative[u1] = sp.sum(numerator/normalizer)

    # Prepare return values, convert sums to means
    totals = {}
    for u,fp in false_positive.iteritems():
        fn = false_negative[u]
        if not spikes[u]:
            totals[u] = (0,0)
        else:
            num = spikes[u]
            totals[u] = (fp / num, fn / num)
    return totals, singles


def calculate_overlap_fp_fn(means, spikes):
    """ Return a dict of tuples (False positive rate, false negative rate)
    indexed by unit.

    .. deprecated:: 0.2.1

    Use :func:`overlap_fp_fn` instead.

    Details for the calculation can be found in
    (Hill et al. The Journal of Neuroscience. 2011). This function works on
    prewhitened data, which means it assumes that all clusters have a uniform
    normal distribution. Data can be prewhitened using the noise covariance
    matrix.

    The calculation for total false positive and false negative rates does
    not follow (Hill et al. The Journal of Neuroscience. 2011), where a
    simple addition of pairwise probabilities is proposed. Instead, the
    total error probabilities are estimated using all clusters at once.

    :param dict means: Dictionary of prewhitened cluster means
        (e.g. unit templates) indexed by unit as :class:`neo.core.Spike`
        objects or numpy arrays for all units.
    :param dict spikes: Dictionary, indexed by unit, of lists of prewhitened
        spike waveforms as :class:`neo.core.Spike` objects or numpy arrays
        for all units.
    :returns: Two values:

        * A dictionary (indexed by unit) of total
          (false positives, false negatives) tuples.
        * A dictionary of dictionaries, both indexed by units,
          of pairwise (false positives, false negatives) tuples.
    :rtype: dict, dict
    """
    units = means.keys()
    if not units:
        return {}, {}

    if len(units) == 1:
        return {units[0]: (0.0, 0.0)}, {}

    # Convert Spike objects to arrays
    spike_arrays = {}
    for u, spks in spikes.iteritems():
        spikelist = []
        for s in spks:
            if isinstance(s, neo.Spike):
                spikelist.append(
                    sp.asarray(s.waveform.rescale(pq.uV)).reshape(-1))
            else:
                spikelist.append(s)
        spike_arrays[u] = sp.asarray(spikelist).T

    # Convert or calculate means
    shaped_means = {}
    for u in units:
        mean = means[u]
        if isinstance(mean, neo.Spike):
            shaped_means[u] = sp.asarray(
                mean.waveform.rescale(pq.uV)).reshape(-1)
        else:
            shaped_means[u] = means[u].reshape(-1)

    return _fast_overlap_whitened(spike_arrays, shaped_means)


def _pair_overlap(waves1, waves2, mean1, mean2, cov1, cov2):
    """ Calculate FP/FN estimates for two gaussian clusters
    """
    from sklearn import mixture

    means = sp.vstack([[mean1], [mean2]])
    covars = sp.vstack([[cov1], [cov2]])
    weights = sp.array([waves1.shape[1], waves2.shape[1]], dtype=float)
    weights /= weights.sum()

    # Create mixture of two Gaussians from the existing estimates
    mix = mixture.GMM(n_components=2, covariance_type='full', init_params='')
    mix.covars_ = covars
    mix.weights_ = weights
    mix.means_ = means

    posterior1 = mix.predict_proba(waves1.T)[:,1]
    posterior2 = mix.predict_proba(waves2.T)[:,0]

    return (posterior1.mean(), posterior2.sum() / len(posterior1),
            posterior2.mean(), posterior1.sum() / len(posterior2))


def overlap_fp_fn(spikes, means=None, covariances=None):
    """ Return dicts of tuples (False positive rate, false negative rate)
    indexed by unit. This function needs :mod:`sklearn` if
    ``covariances`` is not set to ``'white'``.

    This function estimates the pairwise and total false positive and false
    negative rates for a number of waveform clusters. The results can be
    interpreted as follows: False positives are the fraction of spikes in a
    cluster that is estimated to belong to a different cluster (a specific
    cluster for pairwise results or any other cluster for total results).
    False negatives are the number spikes from other clusters that are
    estimated to belong to a given cluster (also expressed as fraction, this
    number can be larger than 1 in extreme cases).

    Details for the calculation can be found in
    (Hill et al. The Journal of Neuroscience. 2011).
    The calculation for total false positive and false negative rates does
    not follow Hill et al., who propose a simple addition of pairwise
    probabilities. Instead, the total error probabilities are estimated
    using all clusters at once.

    :param dict spikes: Dictionary, indexed by unit, of lists of
        spike waveforms as :class:`neo.core.Spike` objects or numpy arrays.
        If the waveforms have multiple channels, they will be reshaped
        automatically. All waveforms need to have the same number of samples.
    :param dict means: Dictionary, indexed by unit, of lists of
        spike waveforms as :class:`neo.core.Spike` objects or numpy arrays.
        Means for units that are not in this dictionary will be estimated
        using the spikes. Note that if you pass ``'white'`` for
        ``covariances`` and you want to provide means, they have to be
        whitened in the same way as the spikes.
        Default: None, means will be estimated from data.
    :param covariances: Dictionary, indexed by unit, of lists of
        covariance matrices. Covariances  for units that are not in this
        dictionary will be estimated using the spikes. It is useful to give
        a covariance matrix if few spikes are present - consider using the
        noise covariance. If you use prewhitened spikes (i.e. all clusters
        are normal distributed, so their covariance matrix is the identity),
        you can pass ``'white'`` here. The calculation will be much faster in
        this case and the sklearn package is not required.
        Default: None, covariances will estimated from data.
    :type covariances: dict or str
    :returns: Two values:

        * A dictionary (indexed by unit) of total
          (false positive rate, false negative rate) tuples.
        * A dictionary of dictionaries, both indexed by units,
          of pairwise (false positive rate, false negative rate) tuples.
    :rtype: dict, dict
    """
    units = spikes.keys()

    total_spikes = 0
    for spks in spikes.itervalues():
        total_spikes += len(spks)
    if total_spikes < 1:
        return {u: (0.0, 0.0) for u in units}, {}

    if means is None:
        means = {}
    white = False
    if covariances is None:
        covariances = {}
    elif covariances == 'white':
        white = True
        covariances = {}

    # Convert Spike objects to arrays
    dimensionality = None
    spike_arrays = {}
    for u, spks in spikes.iteritems():
        spikelist = []
        if not spks or (len(spks) < 2 and u not in covariances):
            units.remove(u)
            continue
        for s in spks:
            if isinstance(s, neo.Spike):
                spikelist.append(
                    sp.asarray(s.waveform.rescale(pq.uV)).reshape(-1))
            else:
                spikelist.append(s)
        spike_arrays[u] = sp.array(spikelist).T
        if dimensionality is None:
            dimensionality = spike_arrays[u].shape[0]
        elif dimensionality != spike_arrays[u].shape[0]:
            raise SpykeException('All spikes need to have the same number'
                                 'of samples!')

    if not units:
        return {}, {}
    if len(units) == 1:
        return {units[0]: (0.0, 0.0)}, {}

    # Convert or calculate means and covariances
    shaped_means = {}
    covs = {}
    if covariances == 'white':
        cov = sp.eye(dimensionality)
        covs = {u:cov for u in units}

    for u in units:
        if u in means:
            mean = means[u]
            if isinstance(mean, neo.Spike):
                shaped_means[u] = sp.asarray(
                    mean.waveform.rescale(pq.uV)).reshape(-1)
            else:
                shaped_means[u] = means[u].reshape(-1)
        else:
            shaped_means[u] = spike_arrays[u].mean(axis=1)

    if white:
        return _fast_overlap_whitened(spike_arrays, shaped_means)

    for u in units:
        if u not in covariances:
            covs[u] = sp.cov(spike_arrays[u])
        else:
            covs[u] = covariances[u]

    # Calculate pairwise false positives/negatives
    singles = {u:{} for u in units}
    for i, u1 in enumerate(units):
        u1 = units[i]
        for u2 in units[i+1:]:
            error_rates = _pair_overlap(spike_arrays[u1], spike_arrays[u2],
                shaped_means[u1], shaped_means[u2], covs[u1], covs[u2])
            singles[u1][u2] = error_rates[0:2]
            singles[u2][u1] = error_rates[2:4]

    # Calculate complete false positives/negatives
    import sklearn
    mix = sklearn.mixture.GMM(n_components=2,
        covariance_type='full')
    mix_means = []
    mix_covars = []
    mix_weights = []
    for u in units:
        mix_means.append(shaped_means[u])
        mix_covars.append([covs[u]])
        mix_weights.append(spike_arrays[u].shape[1])
    mix.means_ = sp.vstack(mix_means)
    mix.covars_ = sp.vstack(mix_covars)
    mix_weights = sp.array(mix_weights, dtype=float)
    mix_weights /= mix_weights.sum()
    mix.weights_ = mix_weights

    # P(spikes of unit[i] in correct cluster)
    post_mean = sp.zeros(len(units))

    # sum(P(spikes of unit[i] in cluster[j])
    post_sum = sp.zeros((len(units), len(units)))

    for i, u in enumerate(units):
        posterior = mix.predict_proba(spike_arrays[u].T)
        post_mean[i] = posterior[:,i].mean()
        post_sum[i,:] = posterior.sum(axis=0)

    totals = {}
    for i, u in enumerate(units):
        fp = 1.0 - post_mean[i]
        ind = range(len(units))
        ind.remove(i)
        fn = post_sum[ind,i].sum() / float(spike_arrays[u].shape[1])
        totals[u] = (fp, fn)

    return totals, singles
