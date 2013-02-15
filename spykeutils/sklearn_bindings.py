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
    def __init__(self, data, metric, tau=1.0 * pq.s):
        self.metric = metric
        self.tau = tau
        self.data = data
        self.x_in = sp.arange(len(data))

    def fit(self, x, y):
        self.gram = metric_defs[self.metric][1](self.data, self.tau)
        self.x_train = x
        return self

    def transform(self, x):
        return self.gram[sp.meshgrid(self.x_train, x)]
