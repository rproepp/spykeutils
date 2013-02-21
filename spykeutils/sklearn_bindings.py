"""
.. data:: metric_defs

    Dictionary of metrics supported by :mod:`.sklearn_bindings`. Each value is
    a tuple consisting of the metrics name and a lambda function taking a list
    of spike trains and a time scale :math:`\\tau` as time quantity.

    The following metrics are available:

    * 'es': :func:`Event synchronization
      <.spike_train_metrics.event_synchronization>` subtracted from 1
    * 'hm': :func:`Hunter-Milton similarity measure
      <.spike_train_metrics.hunter_milton_similarity>` subtracted from 1
    * 'vp': :func:`Victor Purpura's distance
      <.spike_train_metrics.victor_purpura_dist>` with :math:`q = 2/\\tau`
    * 'vr': :func:`Van Rossum distance <.spike_train_metrics.van_rossum_dist>`
"""

import quantities as pq
import scipy as sp
import sklearn.base
import spykeutils.spike_train_metrics as stm


metric_defs = {
    'es': ("Event synchronization",
           lambda trains, tau: 1.0 - stm.event_synchronization(trains, tau)),
    'hm': ("Hunter-Milton similarity measure",
           lambda trains, tau: 1.0 - stm.hunter_milton_similarity(trains, tau)),
    'vp': ("Victor-Purpura\'s distance",
           lambda trains, tau: stm.victor_purpura_dist(trains, 2.0 / tau)),
    'vr': ("Van Rossum distance",
           lambda trains, tau: stm.van_rossum_dist(trains, tau))
}


class PrecomputedSpikeTrainMetricApplier(
        sklearn.base.BaseEstimator, sklearn.base.TransformerMixin):
    """ Precomputes a spike train metric on spike trains and retrieves the
    corresponding Gram matrix (the matrix of all pairwise distances) for
    requested sets of the initial data.

    The spike trains will be passed only once to this class when constructing
    it. At this point the 1D array attribute :attr:`x_in` will be created
    indexing the spike trains. That attribute or a slice of it has then to be
    used is all further method calls requiring input data like :meth:`fit` and
    :meth:`transform`.

    The actual computation of the Gram matrix will be done by calling
    :meth:`fit` with the slice of :attr:`x_in` representing the training set.
    Calls to :meth:`transform` will return subregions out of the precomputed
    Gram matrix.

    .. attribute:: x_in

        A 1D array of indices of the list of spike trains used to initialize the
        instance. Use this or slices of this as input data to method calls
        requiring input data like :meth:`fit` and :meth:`transform` instead of
        the original list of spike trains.
    """

    def __init__(self, data, metric, tau=1.0 * pq.s):
        """
        :param sequence data: A list containing all spike trains which might be
            part of a fitted or transformed set.
        :param str metric: The metric to apply. It has to be a key in
            :const:`metric_defs`. It is not possible to pass a function because
            scikit-learn might pickle the class instance and especially lambda
            functions or class methods cannot be pickled.
        :param tau: Time scale parameter to pass to the metric as time quantity.
        :type tau: Quantity scalar
        """

        self.metric = metric
        self.tau = tau
        self.data = data
        self.x_in = sp.arange(len(data))

    def fit(self, x, y):
        """ Precomputes the Gram matrix.

        :param x: Indices of the training spike trains. Use a slice of
            :attr:`x_in`.
        :type x: 1D array
        :param y: Target labels for classification. Actually unused.
        :returns: Instance on which the method was called.
        :rtype: :class:`PrecomputedSpikeTrainMetricApplier`
        """

        self.gram = metric_defs[self.metric][1](self.data, self.tau)
        self.x_train = x
        return self

    def transform(self, x):
        """ Returns the Gram matrix with the pairwise differences of the
        selected spike trains and the training spike trains.

        :param x: Indices of the spike trains to transform to the Gram matrix.
            Use a slice of :attr:`x_in`.
        :returns: The Gram matrix.
        :rtype: 2D array
        """

        return self.gram[sp.meshgrid(self.x_train, x)]
