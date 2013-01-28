
try:
    import unittest2 as ut
    assert ut  # Suppress pyflakes warning about redefinition of unused ut
except ImportError:
    import unittest as ut

from builders import create_empty_spike_train
import neo
from numpy.testing import assert_array_almost_equal, assert_array_equal
import quantities as pq
import scipy as sp
import spykeutils.signal_processing as sigproc
import spykeutils.spike_train_metrics as stm
import warnings


class CommonMetricTestCases(object):
    """ Provides some common test cases which should work for all spike train
    metrics.
    """

    def calc_metric(self, trains):
        """ Calculates and returns the metric under test.

        :param SpikeTrain a:
        :param SpikeTrain b:
        :rtype: float
        """
        raise NotImplementedError()

    def test_is_zero_for_identical_spike_trains(self):
        st = neo.SpikeTrain(
            sp.array([1, 2, 3]) * pq.s, t_start=0 * pq.s, t_stop=4 * pq.s)
        expected = sp.zeros((2, 2))
        assert_array_almost_equal(expected, self.calc_metric([st, st.copy()]))

    def test_works_with_empty_spike_trains(self):
        st = neo.SpikeTrain(sp.array([]) * pq.s, t_stop=2.0 * pq.s)
        expected = sp.zeros((2, 2))
        assert_array_almost_equal(expected, self.calc_metric([st, st.copy()]))

    def test_is_symmetric(self):
        a = neo.SpikeTrain(sp.array([
            1.1844519,  1.57346687,  2.52261998,  3.65824785,  5.38988771,
            5.63178278,  6.70500182,  7.99562401,  9.21135176
        ]) * pq.s, t_stop=10.0 * pq.s)
        b = neo.SpikeTrain(sp.array([
            0.86096077,  3.54273148,  4.20476326,  6.02451599,  6.42851683,
            6.5564268,  7.07864592,  7.2368936,  7.31784319,  8.15148958,
            8.53540889
        ]) * pq.s, t_stop=10.0 * pq.s)
        assert_array_almost_equal(
            self.calc_metric([a, b]), self.calc_metric([b, a]))


class CommonSimilarityTestCases(object):
    """ Provides some common test cases which should work for all spike train
    similarity measures.
    """

    def calc_similarity(self, trains):
        """ Calculates and returns the similarity measure under test.

        :param SpikeTrain a:
        :param SpikeTrain b:
        :rtype: float
        """
        raise NotImplementedError()

    def test_returns_one_for_equal_spike_trains(self):
        a = neo.SpikeTrain(sp.array([
            1.1844519,  1.57346687,  2.52261998,  3.65824785,  5.38988771,
            5.63178278,  6.70500182,  7.99562401,  9.21135176
        ]) * pq.s, t_stop=10.0 * pq.s)
        expected = sp.ones((2, 2))
        actual = self.calc_similarity([a, a.copy()])
        assert_array_almost_equal(expected, actual)

    def test_is_symmetric(self):
        a = neo.SpikeTrain(sp.array([
            1.1844519,  1.57346687,  2.52261998,  3.65824785,  5.38988771,
            5.63178278,  6.70500182,  7.99562401,  9.21135176
        ]) * pq.s, t_stop=10.0 * pq.s)
        b = neo.SpikeTrain(sp.array([
            0.86096077,  3.54273148,  4.20476326,  6.02451599,  6.42851683,
            6.5564268,  7.07864592,  7.2368936,  7.31784319,  8.15148958,
            8.53540889
        ]) * pq.s, t_stop=10.0 * pq.s)
        assert_array_almost_equal(
            self.calc_similarity([a, b]),
            self.calc_similarity([b, a]))


class Test_cs_dist(ut.TestCase):
    def test_returns_zero_for_equal_spike_trains(self):
        a = neo.SpikeTrain(sp.array([
            1.1844519,  1.57346687,  2.52261998,  3.65824785,  5.38988771,
            5.63178278,  6.70500182,  7.99562401,  9.21135176
        ]) * pq.s, t_stop=10.0 * pq.s, sampling_rate=100 * pq.Hz)
        f = sigproc.GaussianKernel()
        expected = sp.array([[0.0, 0.0], [0.0, 0.0]])
        assert_array_almost_equal(expected, stm.cs_dist(
            [a, a.copy()], f, 1 * pq.Hz))

    def test_returns_nan_if_one_spike_train_is_empty(self):
        empty = create_empty_spike_train()
        non_empty = neo.SpikeTrain(sp.array([1.0]) * pq.s, t_stop=2.0 * pq.s)
        sampling_rate = 100 * pq.Hz
        smoothing_filter = sigproc.GaussianKernel()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            self.assertTrue(sp.all(sp.isnan(stm.cs_dist(
                [empty, non_empty], smoothing_filter,
                sampling_rate=sampling_rate))[(0, 0, 1), (0, 1, 0)]))

    def test_returns_correct_spike_train_cauchy_schwarz_distance(self):
        a = neo.SpikeTrain(
            sp.array([1.0]) * pq.s, t_start=0.6 * pq.s, t_stop=1.4 * pq.s)
        b = neo.SpikeTrain(
            sp.array([0.5, 1.5]) * pq.s, t_stop=2.0 * pq.s)
        c = neo.SpikeTrain(
            sp.array([1.5]) * pq.s, t_start=0.6 * pq.s, t_stop=1.6 * pq.s)
        smoothing_filter = sigproc.GaussianKernel(1.0 * pq.s)
        expected = sp.array(
            [[0.0, 0.12467574, 0.48965132],
            [0.12467574, 0.0, 0.47476452],
            [0.48965132, 0.47476452, 0.0]])
        actual = stm.cs_dist(
            [a, b, c], smoothing_filter, sampling_rate=200 * pq.Hz)
        assert_array_almost_equal(expected, actual, decimal=3)

    def test_is_symmetric(self):
        a = neo.SpikeTrain(sp.array([
            1.1844519,  1.57346687,  2.52261998,  3.65824785,  5.38988771,
            5.63178278,  6.70500182,  7.99562401,  9.21135176
        ]) * pq.s, t_stop=10.0 * pq.s)
        b = neo.SpikeTrain(sp.array([
            0.86096077,  3.54273148,  4.20476326,  6.02451599,  6.42851683,
            6.5564268,  7.07864592,  7.2368936,  7.31784319,  8.15148958,
            8.53540889
        ]) * pq.s, t_stop=10.0 * pq.s)
        f = sigproc.GaussianKernel()
        sampling_rate = 350 * pq.Hz
        assert_array_almost_equal(
            stm.cs_dist([a, b], f, sampling_rate=sampling_rate),
            stm.cs_dist([b, a], f, sampling_rate=sampling_rate), decimal=3)


class Test_event_synchronization(ut.TestCase, CommonSimilarityTestCases):
    def calc_similarity(self, trains):
        return stm.event_synchronization(trains)

    def test_returns_correct_event_synchronization(self):
        a = neo.SpikeTrain(sp.array([1.0, 2.5, 6.5]) * pq.s, t_stop=7.0 * pq.s)
        b = neo.SpikeTrain(sp.array([5.7, 1.0]) * pq.s, t_stop=10.0 * pq.s)
        c = neo.SpikeTrain(sp.array([2.0, 2.1, 5.0]) * pq.s, t_stop=10.0 * pq.s)
        expected = sp.array(
            [[1.0, 0.81649658092772615, 0.0],
             [0.81649658092772615, 1.0, 0.4082482904638631],
             [0.0, 0.4082482904638631, 1.0]])
        actual = stm.event_synchronization([a, b, c])
        assert_array_almost_equal(expected, actual)

    def test_allows_to_set_constant_tau(self):
        a = neo.SpikeTrain(sp.array([1.0, 2.5, 6.5]) * pq.s, t_stop=7.0 * pq.s)
        b = neo.SpikeTrain(sp.array([1.0, 5.7]) * pq.s, t_stop=10.0 * pq.s)
        tau = 0.5 * pq.s
        expected = sp.array(
            [[1.0, 0.40824829046386307],
             [0.40824829046386307, 1.0]])
        actual = stm.event_synchronization([a, b], tau)
        assert_array_almost_equal(expected, actual)

    def test_allows_use_of_different_kernel(self):
        a = neo.SpikeTrain(sp.array([1.0, 2.5, 6.5]) * pq.s, t_stop=7.0 * pq.s)
        b = neo.SpikeTrain(sp.array([1.0, 5.7]) * pq.s, t_stop=10.0 * pq.s)
        tau = 1.0 * pq.s
        kernel = sigproc.LaplacianKernel(1.0, normalize=False)
        expected = sp.array(
            [[1.0, 0.70480122722318095],
             [0.70480122722318095, 1.0]])
        actual = stm.event_synchronization([a, b], tau, kernel=kernel)
        assert_array_almost_equal(expected, actual)


class Test_hunter_milton_similarity(ut.TestCase, CommonSimilarityTestCases):
    def calc_similarity(self, trains):
        return stm.hunter_milton_similarity(trains)

    def test_returns_correct_hunter_milton_similarity(self):
        a = neo.SpikeTrain(sp.array([1.0, 2.5, 6.5]) * pq.s, t_stop=7.0 * pq.s)
        b = neo.SpikeTrain(
            sp.array([1.2, 5.7, 8.0, 9.0]) * pq.s, t_stop=10.0 * pq.s)
        c = neo.SpikeTrain(sp.array([2.1, 2.0, 5.0]) * pq.s, t_stop=10.0 * pq.s)
        tau = 2.0 * pq.s
        expected = sp.array(
            [[1.0, 0.64128747518120299, 0.661254342403672],
             [0.64128747518120299, 1.0, 0.5521235786217787],
             [0.661254342403672, 0.5521235786217787, 1.0]])
        actual = stm.hunter_milton_similarity([a, b, c], tau)
        assert_array_almost_equal(expected, actual)

    def test_allows_use_of_different_kernel(self):
        a = neo.SpikeTrain(sp.array([1.0, 2.5, 6.5]) * pq.s, t_stop=7.0 * pq.s)
        b = neo.SpikeTrain(
            sp.array([1.2, 5.7, 8.0, 9.0]) * pq.s, t_stop=10.0 * pq.s)
        kernel = sigproc.TriangularKernel(1.0 * pq.s, normalize=False)
        expected = sp.array(
            [[1.0, 0.29166666666666663], [0.29166666666666663, 1.0]])
        actual = stm.hunter_milton_similarity([a, b], kernel=kernel)
        assert_array_almost_equal(expected, actual)

    def test_spike_trains_may_be_empty(self):
        empty = create_empty_spike_train()
        non_empty = neo.SpikeTrain(sp.array([1.0]) * pq.s, t_stop=3.0 * pq.s)
        expected = sp.array([[1.0, 0.0], [0.0, 1.0]])
        actual = stm.hunter_milton_similarity([empty, non_empty])
        assert_array_almost_equal(expected, actual)


class Test_norm_dist(ut.TestCase):
    def test_returns_zero_for_equal_spike_trains(self):
        st = neo.SpikeTrain(sp.array([
            1.1844519,  1.57346687,  2.52261998,  3.65824785,  5.38988771,
            5.63178278,  6.70500182,  7.99562401,  9.21135176
        ]) * pq.s, t_stop=10.0 * pq.s, sampling_rate=100 * pq.Hz)
        f = sigproc.GaussianKernel()
        expected = sp.zeros((2, 2)) * pq.Hz ** 0.5
        assert_array_almost_equal(expected, stm.norm_dist(
            [st, st.copy()], f, 1 * pq.Hz))

    def test_returns_norm_if_one_spike_train_is_empty(self):
        empty = create_empty_spike_train()
        non_empty = neo.SpikeTrain(sp.array([1.0]) * pq.s, t_stop=2.0 * pq.s)
        sampling_rate = 100 * pq.Hz
        smoothing_filter = sigproc.GaussianKernel()
        norm = stm.st_norm(
            non_empty, smoothing_filter, sampling_rate=sampling_rate)
        expected = sp.array([[0.0, norm], [norm, 0.0]]) * pq.Hz ** 0.5
        actual = stm.norm_dist(
            [empty, non_empty], smoothing_filter, sampling_rate=sampling_rate)
        assert_array_almost_equal(expected, actual, decimal=3)

    def test_returns_correct_spike_train_norm_distance(self):
        a = neo.SpikeTrain(
            sp.array([1.0]) * pq.s, t_start=0.6 * pq.s, t_stop=1.4 * pq.s)
        b = neo.SpikeTrain(sp.array([0.5, 1.5]) * pq.s, t_stop=2.0 * pq.s)
        c = neo.SpikeTrain(sp.array([1.0, 2.0]) * pq.s, t_stop=3.0 * pq.s)
        smoothing_filter = sigproc.GaussianKernel(1.0 * pq.s)
        expected = sp.array(
            [[0.0, 0.475035, 0.531116],
             [0.475035, 0.0, 0.309422],
             [0.531116, 0.309422, 0.0]]) * pq.Hz ** 0.5
        actual = stm.norm_dist(
            [a, b, c], smoothing_filter, sampling_rate=200 * pq.Hz)
        assert_array_almost_equal(
            expected, actual.rescale(expected.units), decimal=3)

    def test_is_symmetric(self):
        a = neo.SpikeTrain(sp.array([
            1.1844519,  1.57346687,  2.52261998,  3.65824785,  5.38988771,
            5.63178278,  6.70500182,  7.99562401,  9.21135176
        ]) * pq.s, t_stop=10.0 * pq.s)
        b = neo.SpikeTrain(sp.array([
            0.86096077,  3.54273148,  4.20476326,  6.02451599,  6.42851683,
            6.5564268,  7.07864592,  7.2368936,  7.31784319,  8.15148958,
            8.53540889
        ]) * pq.s, t_stop=10.0 * pq.s)
        f = sigproc.GaussianKernel()
        sampling_rate = 350 * pq.Hz
        assert_array_almost_equal(
            stm.norm_dist([a, b], f, sampling_rate=sampling_rate),
            stm.norm_dist([b, a], f, sampling_rate=sampling_rate), decimal=3)


class Test_schreiber_similarity(ut.TestCase, CommonSimilarityTestCases):
    def calc_similarity(self, trains):
        k = sigproc.GaussianKernel()
        return stm.schreiber_similarity(trains, k)

    def test_returns_nan_if_one_spike_train_is_empty(self):
        empty = create_empty_spike_train()
        non_empty = neo.SpikeTrain(sp.array([1.0]) * pq.s, t_stop=2.0 * pq.s)
        k = sigproc.GaussianKernel()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            actual = stm.schreiber_similarity((empty, non_empty), k)
        self.assertTrue(sp.isnan(actual[0, 0]))
        self.assertTrue(sp.isnan(actual[0, 1]))
        self.assertTrue(sp.isnan(actual[1, 0]))

    def test_returns_correct_spike_train_schreiber_similarity(self):
        a = neo.SpikeTrain(
            sp.array([1.0]) * pq.s, t_start=0.6 * pq.s, t_stop=1.4 * pq.s)
        b = neo.SpikeTrain(
            sp.array([0.5, 1.5]) * pq.s, t_stop=2.0 * pq.s)
        c = neo.SpikeTrain(
            sp.array([2.0, 1.0]) * pq.s, t_start=0.6 * pq.s, t_stop=2.4 * pq.s)
        k = sigproc.GaussianKernel(sp.sqrt(2.0) * pq.s)
        expected = sp.array([
            [1.0, 0.9961114, 0.9430803],
            [0.9961114, 1.0, 0.9523332],
            [0.9430803, 0.9523332, 1.0]])
        actual = stm.schreiber_similarity((a, b, c), k)
        assert_array_almost_equal(expected, actual)


class Test_st_inner(ut.TestCase):
    def test_returns_zero_if_any_spike_train_is_empty(self):
        empty = create_empty_spike_train()
        non_empty = neo.SpikeTrain(sp.array([1.0]) * pq.s, t_stop=2.0 * pq.s)
        smoothing_filter = sigproc.GaussianKernel()
        sampling_rate = 1 * pq.Hz
        expected = sp.array([0.0]) * pq.Hz
        self.assertAlmostEqual(
            expected, stm.st_inner(
                [empty], [empty], smoothing_filter, sampling_rate))
        self.assertAlmostEqual(
            expected, stm.st_inner(
                [empty], [non_empty], smoothing_filter, sampling_rate))
        self.assertAlmostEqual(
            expected, stm.st_inner(
                [non_empty], [empty], smoothing_filter, sampling_rate))

    def test_returns_correct_inner_spike_train_product(self):
        a = neo.SpikeTrain(
            sp.array([1.0]) * pq.s, t_start=0.6 * pq.s, t_stop=1.4 * pq.s)
        b = neo.SpikeTrain(
            sp.array([0.5, 1.5]) * pq.s, t_stop=2.0 * pq.s)
        smoothing_filter = sigproc.GaussianKernel(1.0 * pq.s)
        expected = 0.530007 * pq.Hz
        actual = stm.st_inner(
            [a], [b], smoothing_filter, sampling_rate=100 * pq.Hz)
        self.assertAlmostEqual(
            expected, actual.rescale(expected.units), places=3)

    def test_is_symmetric(self):
        a = neo.SpikeTrain(sp.array([
            1.1844519,  1.57346687,  2.52261998,  3.65824785,  5.38988771,
            5.63178278,  6.70500182,  7.99562401,  9.21135176
        ]) * pq.s, t_stop=10.0 * pq.s)
        b = neo.SpikeTrain(sp.array([
            0.86096077,  3.54273148,  4.20476326,  6.02451599,  6.42851683,
            6.5564268,  7.07864592,  7.2368936,  7.31784319,  8.15148958,
            8.53540889
        ]) * pq.s, t_stop=10.0 * pq.s)
        f = sigproc.GaussianKernel()
        sampling_rate = 150 * pq.Hz
        assert_array_almost_equal(
            stm.st_inner([a], [b], f, sampling_rate=sampling_rate),
            stm.st_inner([b], [a], f, sampling_rate=sampling_rate), decimal=3)

    def test_accepts_sequences_of_spike_trains(self):
        a = neo.SpikeTrain(
            sp.array([1000.0]) * pq.ms, t_start=0.6 * pq.s, t_stop=1.4 * pq.s)
        b = neo.SpikeTrain(
            sp.array([0.5, 1.5]) * pq.s, t_stop=2.0 * pq.s)
        f = sigproc.GaussianKernel()
        sampling_rate = 150 * pq.Hz
        expected = sp.array(
            [[0.282094, 0.530072], [0.530072, 1.003787]]) * pq.Hz
        actual = stm.st_inner([a, b], [a, b], f, sampling_rate=sampling_rate)
        assert_array_almost_equal(expected, actual, decimal=3)


class Test_st_norm(ut.TestCase):
    def test_returns_zero_if_spike_train_is_empty(self):
        empty = create_empty_spike_train()
        smoothing_filter = sigproc.GaussianKernel()
        self.assertAlmostEqual(0.0, stm.st_norm(
            empty, smoothing_filter, 1 * pq.Hz))

    def test_returns_correct_spike_train_norm(self):
        st = neo.SpikeTrain(
            sp.array([0.5, 1.0, 1.5]) * pq.s, t_stop=2.0 * pq.s)
        smoothing_filter = sigproc.GaussianKernel(1.0 * pq.s)
        expected = (2.34569 * pq.Hz) ** 0.5
        actual = stm.st_norm(st, smoothing_filter, sampling_rate=200 * pq.Hz)
        self.assertAlmostEqual(
            expected, actual.rescale(expected.units), places=3)


class Test_van_rossum_dist(ut.TestCase, CommonMetricTestCases):
    def calc_metric(self, trains):
        return stm.van_rossum_dist(trains)

    def test_return_correct_distance(self):
        a = neo.SpikeTrain(
            sp.array([1.0, 4.0, 5.0, 6.0, 9.0, 11.0]) * pq.s,
            t_stop=12.0 * pq.s)
        b = neo.SpikeTrain(
            sp.array([2.0, 4.0, 7.0, 10.0]) * pq.s,  t_stop=12.0 * pq.s)
        c = neo.SpikeTrain(sp.array([4.0, 3.0]) * pq.s, t_stop=12.0 * pq.s)
        tau = 3.0 * pq.s
        expected = sp.array([
            [0.0, 1.895846644204, 2.878796160479],
            [1.895846644204, 0.0, 1.760192079676],
            [2.878796160479, 1.760192079676, 0.0]])
        actual = stm.van_rossum_dist((a, b, c), tau)
        assert_array_almost_equal(expected, actual)

    def test_distance_of_empty_spiketrain_and_single_spike_equals_one(self):
        a = neo.SpikeTrain(sp.array([]) * pq.s, t_stop=2.0 * pq.s)
        b = neo.SpikeTrain(sp.array([1.0]) * pq.s, t_stop=2.0 * pq.s)
        expected = sp.array([[0.0, 1.0], [1.0, 0.0]])
        actual = stm.van_rossum_dist((a, b), 3.0 * pq.s)
        assert_array_almost_equal(expected, actual)

    def test_allows_use_of_different_kernel(self):
        a = neo.SpikeTrain(sp.array([1.0, 2.0]) * pq.s, t_stop=3.0 * pq.s)
        b = neo.SpikeTrain(sp.array([1.5]) * pq.s, t_stop=2.0 * pq.s)
        k = sigproc.GaussianKernel(1.0 * pq.s, normalize=False)
        expected = sp.array([
            [0.0, 0.8264827],
            [0.8264827, 0.0]])
        actual = stm.van_rossum_dist((a, b), kernel=k)
        assert_array_almost_equal(expected, actual)

    def test_allows_tau_equal_to_infinity(self):
        a = neo.SpikeTrain(sp.array([1.0, 1.9, 2.0]) * pq.s, t_stop=3.0 * pq.s)
        b = neo.SpikeTrain(sp.array([1.5]) * pq.s, t_stop=2.0 * pq.s)
        tau = sp.inf * pq.s
        expected = sp.array([
            [0.0, 2.0],
            [2.0, 0.0]])
        actual = stm.van_rossum_dist((a, b), tau)
        assert_array_almost_equal(expected, actual)


class Test_van_rossum_multiunit_dist(ut.TestCase, CommonMetricTestCases):
    # With only one spike train each we should get the normal van Rossum
    # distance.
    def calc_metric(self, trains):
        return stm.van_rossum_multiunit_dist({0: trains}, 1)

    def test_returns_correct_distance_for_multiunits(self):
        a0 = neo.SpikeTrain(sp.array([1.0, 5.0, 7.0]) * pq.s, t_stop=8.0 * pq.s)
        a1 = neo.SpikeTrain(sp.array([1.0, 2.0, 5.0]) * pq.s, t_stop=8.0 * pq.s)
        b0 = neo.SpikeTrain(sp.array([2.0, 4.0, 5.0]) * pq.s, t_stop=8.0 * pq.s)
        b1 = neo.SpikeTrain(sp.array([3.0, 8.0]) * pq.s, t_stop=9.0 * pq.s)
        units = {0: [a0, a1], 1: [b0, b1]}
        weighting = 0.3
        expected = sp.array([[0.0, 2.37006181], [2.37006181, 0.0]])
        actual = stm.van_rossum_multiunit_dist(units, weighting)
        assert_array_almost_equal(expected, actual)

    def test_allows_tau_equal_to_infinity_with_multiunits(self):
        a0 = neo.SpikeTrain(sp.array([1.0, 5.0, 7.0]) * pq.s, t_stop=8.0 * pq.s)
        a1 = neo.SpikeTrain(sp.array([5.0]) * pq.s, t_stop=8.0 * pq.s)
        b0 = neo.SpikeTrain(sp.array([2.0, 4.0, 5.0]) * pq.s, t_stop=8.0 * pq.s)
        b1 = neo.SpikeTrain(sp.array([3.0, 8.0]) * pq.s, t_stop=9.0 * pq.s)
        units = {0: [a0, a1], 1: [b0, b1]}
        weighting = 0.3
        tau = sp.inf * pq.s
        dist = sp.sqrt(5.0 + weighting * 4.0)
        expected = sp.array([[0.0, dist], [dist, 0.0]])
        actual = stm.van_rossum_multiunit_dist(units, weighting, tau)
        assert_array_almost_equal(expected, actual)


class Test_victor_purpura_dist(ut.TestCase, CommonMetricTestCases):
    def calc_metric(self, trains):
        return stm.victor_purpura_dist(trains)

    def test_inserted_spikes_equal_cost_of_one(self):
        num_spikes = 3
        st = neo.SpikeTrain(sp.arange(3) * pq.s, t_stop=num_spikes * pq.s)
        st_empty = create_empty_spike_train()
        expected = sp.array([[0.0, num_spikes], [num_spikes, 0.0]])
        assert_array_almost_equal(
            expected, stm.victor_purpura_dist([st, st_empty]))

    def test_returns_q_weighted_dist_for_close_spike_pair(self):
        a = neo.SpikeTrain(sp.array([1.0]) * pq.s, t_stop=2 * pq.s)
        b = neo.SpikeTrain(sp.array([1.5]) * pq.s, t_stop=2 * pq.s)
        q = 2.0 / pq.s
        expected = sp.array([[0.0, 0.5 * 2.0], [0.5 * 2.0, 0.0]])
        assert_array_almost_equal(expected, stm.victor_purpura_dist([a, b], q))

    def test_returns_two_for_distant_spike_pair(self):
        a = neo.SpikeTrain(sp.array([1.0]) * pq.s, t_stop=6 * pq.s)
        b = neo.SpikeTrain(sp.array([5.0]) * pq.s, t_stop=6 * pq.s)
        q = 1.0 / pq.s
        expected = sp.array([[0.0, 2.0], [2.0, 0.0]])
        assert_array_almost_equal(expected, stm.victor_purpura_dist([a, b], q))

    def test_returns_correct_distance_for_two_spike_trains(self):
        q = 1.0 / pq.s
        a = neo.SpikeTrain(
            sp.array([1.0, 2.0, 4.1, 7.0, 7.1]) * pq.s, t_stop=8.0 * pq.s)
        b = neo.SpikeTrain(
            sp.array([1.2, 4.0, 4.3]) * pq.s, t_stop=8.0 * pq.s)
        # From a to b:
        #   - shift 1.0 to 1.2 (cost 0.2)
        #   - delete 2.0 (cost 1.0)
        #   - shift 4.1 to 4.0 (cost 0.1)
        #   - insert 4.3 (cost 1.0)
        #   - delete 7.0 (cost 1.0)
        #   - delete 7.1 (cost 1.0)
        expected = sp.array([[0.0, 4.3], [4.3, 0.0]])
        assert_array_almost_equal(expected, stm.victor_purpura_dist([a, b], q))

    def test_allows_use_of_different_kernel(self):
        k = sigproc.LaplacianKernel(1.0 * pq.s, normalize=False)
        a = neo.SpikeTrain(
            sp.array([1.0, 2.0, 4.1, 7.0, 7.1]) * pq.s, t_stop=8.0 * pq.s)
        b = neo.SpikeTrain(
            sp.array([4.0, 4.3, 1.2]) * pq.s, t_stop=8.0 * pq.s)
        # From a to b:
        #   - shift 1.0 to 1.2 (cost 0.3625385)
        #   - delete 2.0 (cost 1.0)
        #   - shift 4.1 to 4.0 (cost 0.1903252)
        #   - shift 4.3 to 7.0 (cost 1.8655890)
        #   - delete 7.0 (cost 1.0)
        #   - delete 7.1 (cost 1.0)
        expected = sp.array([[0.0, 4.4184526], [4.4184526, 0.0]])
        assert_array_almost_equal(
            expected, stm.victor_purpura_dist([a, b], kernel=k))


class Test_victor_purpura_multiunit_dist(ut.TestCase, CommonMetricTestCases):
    # With only one spike train each we should get the normal VP distance.
    def calc_metric(self, trains):
        return stm.victor_purpura_multiunit_dist({0: trains}, 1)

    def test_returns_correct_distance_for_multiunits(self):
        a0 = neo.SpikeTrain(sp.array([1.0, 5.0, 7.0]) * pq.s, t_stop=8.0 * pq.s)
        a1 = neo.SpikeTrain(sp.array([1.0, 2.0, 5.0]) * pq.s, t_stop=8.0 * pq.s)
        b0 = neo.SpikeTrain(sp.array([2.0, 4.0, 5.0]) * pq.s, t_stop=8.0 * pq.s)
        b1 = neo.SpikeTrain(sp.array([3.0, 8.0]) * pq.s, t_stop=9.0 * pq.s)
        units = {0: [a0, a1], 1: [b0, b1]}
        reassignment_cost = 0.7
        expected = sp.array([[0.0, 4.4], [4.4, 0.0]])
        actual = stm.victor_purpura_multiunit_dist(units, reassignment_cost)
        assert_array_almost_equal(expected, actual)

    def test_returns_empty_array_if_empty_dict_is_passed(self):
        expected = sp.zeros((0, 0))
        actual = stm.victor_purpura_multiunit_dist({}, 1.0)
        assert_array_equal(expected, actual)

    def test_returns_empty_array_if_trials_are_empty(self):
        expected = sp.zeros((0, 0))
        actual = stm.victor_purpura_multiunit_dist({0: [], 1: []}, 1.0)
        assert_array_equal(expected, actual)

    def test_raises_exception_if_number_of_trials_differs(self):
        st = create_empty_spike_train()
        with self.assertRaises(ValueError):
            stm.victor_purpura_multiunit_dist({0: [st], 1: [st, st]}, 1.0)


if __name__ == '__main__':
    ut.main()
