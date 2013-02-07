
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
            [a], [b], smoothing_filter, sampling_rate=1000 * pq.Hz)
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

    def test_does_not_fail_with_kernel_not_allowing_spike_trains_as_argument(
            self):
        # Compare <https://neuralensemble.org/trac/neo/ticket/65>
        a = neo.SpikeTrain(sp.array([1.0, 2.0]) * pq.s, t_stop=3.0 * pq.s)
        b = neo.SpikeTrain(sp.array([1.5]) * pq.s, t_stop=2.0 * pq.s)
        k = sigproc.TriangularKernel(1.0 * pq.s, normalize=False)
        stm.van_rossum_dist((a, b), kernel=k)

    def test_allows_tau_equal_to_infinity(self):
        a = neo.SpikeTrain(sp.array([1.0, 1.9, 2.0]) * pq.s, t_stop=3.0 * pq.s)
        b = neo.SpikeTrain(sp.array([1.5]) * pq.s, t_stop=2.0 * pq.s)
        tau = sp.inf * pq.s
        expected = sp.array([
            [0.0, 4.0],
            [4.0, 0.0]])
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

    def test_returns_correct_distance_for_complex_spike_trains(self):
        # This is a regression test for a bug that certain spike trains where
        # producing to large distances.
        trains = [
            neo.SpikeTrain(sp.array([
                0.02675798, 0.03181146, 0.03341735, 0.03775562, 0.07791623,
                0.08822388, 0.10770132, 0.12325048, 0.16989942, 0.2017788,
                0.20671708, 0.21338806, 0.24211925, 0.25483266, 0.27496442,
                0.27587779, 0.27987714, 0.29092447, 0.3126856, 0.31699044,
                0.33125793, 0.38880785, 0.38881775, 0.44730422, 0.47123718,
                0.47530894, 0.50035773, 0.5110994, 0.5406418, 0.55932289,
                0.56299461, 0.61127646, 0.6669967, 0.6878365, 0.69095517,
                0.71292938, 0.74403481, 0.79566084, 0.80520382, 0.87465267,
                0.9233359, 0.97119188, 0.97221954, 0.98573419, 1.00598374,
                1.08840599, 1.10346633, 1.11300801, 1.11736787, 1.17619865,
                1.17750093, 1.18119904, 1.19001107, 1.23349135, 1.24515837,
                1.31601168, 1.32058585, 1.3274779, 1.3304611, 1.39192936,
                1.396939, 1.42214471, 1.43682422, 1.44056841, 1.44614004,
                1.45396973, 1.48746414, 1.51381587, 1.52504075, 1.56534678,
                1.56654466, 1.56932347, 1.62405807, 1.63243667, 1.64011958,
                1.65348796, 1.67166925, 1.6899014, 1.70019229, 1.71489787,
                1.7498802, 1.75506253, 1.77316786, 1.79721912, 1.80556803,
                1.82943579, 1.8393378, 1.85571875, 1.86451301, 1.86915057,
                1.93494862, 1.95227868, 1.95787129, 2.01151238, 2.05108779,
                2.05622847, 2.07756536, 2.09751716, 2.11014462, 2.12756709,
                2.1301002, 2.22850943, 2.23546736, 2.26357638, 2.32916089,
                2.35222596, 2.36019072, 2.44110203, 2.48733729, 2.48868378,
                2.49587805, 2.50484364, 2.52888902, 2.54460952, 2.55477246,
                2.56718557, 2.57197204, 2.58715912, 2.62834212, 2.6607554,
                2.71456005, 2.71990732, 2.73476721, 2.76560221, 2.79578411,
                2.81420671, 2.82842414, 2.84323564, 2.86808335, 2.89346033,
                2.89759722, 2.90250757, 2.92396906, 2.96089258, 2.99343156,
                2.99513297, 3.00295214, 3.00404354, 3.01155098, 3.0220984,
                3.06860675, 3.10776003, 3.11125211, 3.12200107, 3.13822244,
                3.172325, 3.17359243, 3.17693368, 3.18779785, 3.1898421,
                3.2027296, 3.20308197, 3.22950711, 3.23480067, 3.25230996,
                3.26324005, 3.30303045, 3.3323502, 3.34200826, 3.38362587,
                3.39374602, 3.40100303, 3.42572902, 3.43405842, 3.48714745,
                3.48808569, 3.50765539, 3.51019425, 3.51654164, 3.53508831,
                3.55091076, 3.55806575, 3.56160866, 3.64616879, 3.66336828,
                3.70684962, 3.71508041, 3.71960502, 3.74088608, 3.7694215,
                3.78687648, 3.78826898, 3.8032681, 3.80442445, 3.82610046,
                3.83252045, 3.83375399, 3.83963007, 3.87070708, 3.89921058,
                3.91373461, 3.98189025, 3.99281868, 3.99615101, 4.03866165,
                4.06296107, 4.0664576, 4.10008341, 4.13249147, 4.14948245,
                4.15544816, 4.18645968, 4.23956819, 4.24159763, 4.25682634,
                4.29096996, 4.29801235, 4.30704865, 4.3364981, 4.34955189,
                4.35691426, 4.37946289, 4.40449102, 4.41415224, 4.42969554,
                4.43297123, 4.43672311, 4.46269914, 4.50611436, 4.54325245,
                4.59867291, 4.6118659, 4.62881441, 4.64220816, 4.68635809,
                4.6919799, 4.69224906, 4.71150593, 4.75981344, 4.76055566,
                4.8129406, 4.83692968, 4.87459801, 4.8895905, 4.89048346,
                4.90390866, 4.92131202, 4.95374717, 4.95658542, 4.9779478,
                4.99927772, 5.00321623, 5.02279036, 5.02980636, 5.06479496,
                5.07471904, 5.09194692, 5.1019829, 5.10353541, 5.10481109,
                5.10639536, 5.13999128, 5.1501336, 5.15279668, 5.16864755,
                5.18043833, 5.18738265, 5.19297201, 5.19496814, 5.19888859,
                5.20398454, 5.24268102, 5.25293838, 5.25735102, 5.27904209,
                5.32513061, 5.33412059, 5.35300406, 5.36359518, 5.38220169,
                5.41380451, 5.44608516, 5.45664259, 5.46624451, 5.49995728,
                5.52030155, 5.52986433, 5.53527111, 5.58813843, 5.5986904,
                5.63867497, 5.64965832, 5.70854657, 5.77092465, 5.78018575,
                5.80469618, 5.82611303, 5.84211921, 5.84769114, 5.85898366,
                5.86174668, 5.86686434, 5.86807339, 5.88557362, 5.93531383,
                5.94590946, 5.9535614, 5.96181496, 5.96211509, 5.96322495,
                5.99951691, 6.02956462, 6.03071066, 6.11325118, 6.12068097,
                6.13916618, 6.15618799, 6.17405661, 6.19074313, 6.20637448,
                6.21144991, 6.22694995, 6.2504859, 6.29414487, 6.3132762,
                6.37532399, 6.37625784, 6.41398007, 6.41816266, 6.42386713,
                6.42767342, 6.43909112, 6.48312163, 6.50112821, 6.50284644,
                6.52335736, 6.55053573, 6.55945474, 6.56113336, 6.58452909,
                6.58510608, 6.59753607, 6.61954437, 6.64973018, 6.66495931,
                6.66904812, 6.67276565, 6.73951848, 6.75443413, 6.75483586,
                6.79528155, 6.81670372, 6.83292695, 6.84892368, 6.90221611,
                6.94186031, 6.97372169, 6.98930105, 7.00503816, 7.01156979,
                7.01622253, 7.04066381, 7.08116801, 7.1022431, 7.10534942,
                7.12276162, 7.17072979, 7.1846351, 7.21250037, 7.23569895,
                7.23759221, 7.26638189, 7.31573003, 7.39632157, 7.40696688,
                7.42971144, 7.45062847, 7.4634739, 7.4718392, 7.49271328,
                7.55204862, 7.59257437, 7.60799196, 7.61363934, 7.62867287,
                7.64457945, 7.65194936, 7.66110909, 7.66676376, 7.67758238,
                7.68405278, 7.69391715, 7.6990212, 7.72407479, 7.75592843,
                7.77321337, 7.78914379, 7.80573035, 7.81001852, 7.81201576,
                7.81761754, 7.822486, 7.88454532, 7.90159693, 7.92447452,
                7.93032758, 7.95127432, 7.95471672, 7.95611181, 7.99765534,
                8.00169997, 8.05611102, 8.06999799, 8.0877689, 8.11370158,
                8.12326905, 8.19558094, 8.20785861, 8.22790536, 8.25096989,
                8.29404755, 8.32625888, 8.38768653, 8.41293726, 8.44072146,
                8.45655928, 8.46028366, 8.46062243, 8.47631889, 8.50685359,
                8.539859, 8.55656747, 8.57298557, 8.60573667, 8.65462893,
                8.67784071, 8.68571095, 8.71909035, 8.72206184, 8.7314385,
                8.73608901, 8.74239948, 8.74416149, 8.75145957, 8.77516598,
                8.88377333, 8.8848043, 8.89789711, 8.91243437, 8.91476806,
                8.91492797, 8.92139551, 8.93704381, 8.96318634, 8.99623903,
                9.00131449, 9.01417633, 9.01421952, 9.03203569, 9.03786051,
                9.04157583, 9.09361684, 9.09610771, 9.10131371, 9.10609705,
                9.12084572, 9.15575811, 9.15644013, 9.1691256, 9.18362837,
                9.18595479, 9.21164258, 9.24095542, 9.24290778, 9.25767234,
                9.26005027, 9.26048416, 9.28017441, 9.29182669, 9.30192562,
                9.31486222, 9.35580549, 9.37514957, 9.43470264, 9.46401276,
                9.48844607, 9.4945491, 9.50132042, 9.5133463, 9.51426077,
                9.52668188, 9.52888838, 9.53854506, 9.54400945, 9.55057675,
                9.57993589, 9.63604947, 9.64316243, 9.66791914, 9.70282942,
                9.71906419, 9.72696098, 9.7422066, 9.74416635, 9.76302569,
                9.77237119, 9.77808876, 9.78865054, 9.79208195, 9.82398648,
                9.83977829, 9.85440184, 9.87001817, 9.91401035, 9.92363489,
                9.9347058, 9.94121602, 9.95317336, 9.95549832, 9.95695226,
                9.97754868, 9.98384015]) * pq.s, t_stop=10.0 * pq.s),
            neo.SpikeTrain(sp.array([
                0.0114491, 0.02651815, 0.02672949, 0.02712123, 0.03514833,
                0.05414386, 0.07570339, 0.09427385, 0.10903071, 0.11588711,
                0.11739125, 0.1285715, 0.14934368, 0.16684372, 0.21166201,
                0.22235881, 0.23386214, 0.24181703, 0.25805984, 0.2654033,
                0.27348522, 0.30116999, 0.31207604, 0.31553495, 0.32936142,
                0.32953416, 0.35437639, 0.40074384, 0.41165687, 0.44861386,
                0.49870305, 0.5265349, 0.53879183, 0.57395557, 0.62112778,
                0.63952386, 0.65174804, 0.68523672, 0.72572932, 0.74491922,
                0.77281653, 0.77533443, 0.83372669, 0.84671895, 0.87093241,
                0.92558636, 0.94601541, 0.94777018, 0.94821996, 0.97271642,
                1.0005331, 1.00257254, 1.00735428, 1.0198866, 1.04727644,
                1.09182491, 1.09894488, 1.10078114, 1.10360265, 1.11904421,
                1.12918186, 1.13765565, 1.18229212, 1.20436513, 1.21081849,
                1.22066808, 1.22314962, 1.26854532, 1.30229203, 1.31703206,
                1.32135388, 1.32907158, 1.33047318, 1.36227875, 1.39697511,
                1.4242654, 1.4244518, 1.43681519, 1.4493789, 1.45152151,
                1.46461455, 1.47733094, 1.48771515, 1.53536739, 1.54864524,
                1.55283995, 1.5898638, 1.60887471, 1.64490284, 1.64502768,
                1.66872741, 1.70025134, 1.71529419, 1.71851586, 1.75573609,
                1.78231052, 1.8083983, 1.81541951, 1.81772587, 1.84818917,
                1.85059323, 1.88875683, 1.90898902, 1.93557862, 1.9643203,
                1.96710505, 1.98391057, 1.98527593, 2.03665079, 2.08708411,
                2.08761721, 2.11103023, 2.12101666, 2.13992148, 2.17117369,
                2.18684568, 2.22655021, 2.24875486, 2.24929527, 2.28056109,
                2.28729401, 2.31258209, 2.32301025, 2.32477238, 2.32491974,
                2.34173467, 2.35126611, 2.35149399, 2.38431406, 2.40687869,
                2.42583741, 2.42797991, 2.42828893, 2.45838911, 2.46432188,
                2.46473762, 2.47316229, 2.51085401, 2.5283335, 2.55848724,
                2.56442768, 2.59182815, 2.60989243, 2.65008826, 2.67778032,
                2.67781156, 2.68312729, 2.68929609, 2.70518959, 2.73459435,
                2.78244226, 2.78290087, 2.79595168, 2.80616739, 2.80701334,
                2.81042141, 2.85470512, 2.87509772, 2.88886327, 2.89375791,
                2.97284058, 2.97512514, 2.98540772, 3.01458122, 3.03159057,
                3.05350786, 3.05518717, 3.10446297, 3.13792582, 3.15204826,
                3.17267234, 3.19586531, 3.19657011, 3.21282816, 3.25677248,
                3.27720176, 3.28887985, 3.29735282, 3.2982325, 3.32269346,
                3.32343112, 3.32637092, 3.34520261, 3.34914751, 3.4176678,
                3.43099532, 3.48336162, 3.48518715, 3.52127749, 3.52151362,
                3.5773688, 3.59222194, 3.6013162, 3.62748155, 3.63613575,
                3.64713969, 3.65456465, 3.66853991, 3.73818958, 3.74375182,
                3.80164474, 3.86614106, 3.89385381, 3.97585319, 3.98647681,
                4.00558264, 4.0212778, 4.05202117, 4.0594387, 4.09760178,
                4.11367539, 4.12070204, 4.12999226, 4.15656723, 4.20514307,
                4.27451413, 4.27635573, 4.28445258, 4.28533623, 4.33012486,
                4.35620149, 4.37670464, 4.37681744, 4.39893272, 4.44981225,
                4.45885746, 4.47979453, 4.48028014, 4.51009319, 4.52546144,
                4.57879502, 4.66509915, 4.71338549, 4.71713202, 4.73567885,
                4.75441602, 4.79556635, 4.79582663, 4.82047298, 4.82055109,
                4.83059559, 4.83590133, 4.86399401, 4.87413277, 4.87833755,
                4.89208783, 4.9192821, 4.941063, 4.98772884, 5.01993596,
                5.02465223, 5.06293715, 5.06939498, 5.07198031, 5.11089343,
                5.14112836, 5.15388206, 5.18105507, 5.19314929, 5.19670734,
                5.22545792, 5.23334406, 5.23459961, 5.2494979, 5.2573258,
                5.25908266, 5.2840583, 5.2853253, 5.28590158, 5.32314432,
                5.35959824, 5.36241399, 5.38921977, 5.40694111, 5.4313708,
                5.46598325, 5.47254526, 5.49387086, 5.49886878, 5.56592236,
                5.57180461, 5.58869339, 5.58984367, 5.59601824, 5.62938579,
                5.64426059, 5.6476461, 5.67241871, 5.6771723, 5.67873946,
                5.68074113, 5.72312447, 5.7271727, 5.76271693, 5.79335885,
                5.80349046, 5.83560725, 5.84101573, 5.85666574, 5.8643614,
                5.86509986, 5.86531037, 5.87744489, 5.90506991, 5.91776312,
                5.96371983, 5.96613482, 5.98032448, 5.98608614, 6.00144331,
                6.00838531, 6.00846468, 6.01048934, 6.02474142, 6.0335397,
                6.05113466, 6.06459963, 6.06576204, 6.08503265, 6.10602749,
                6.10606072, 6.22065498, 6.2532318, 6.29605114, 6.31945753,
                6.35632236, 6.35896878, 6.36120413, 6.38709957, 6.39295197,
                6.41809868, 6.42367352, 6.44628183, 6.47049815, 6.48133661,
                6.49090302, 6.49289679, 6.50896993, 6.51693538, 6.54015486,
                6.56308082, 6.568914, 6.57395747, 6.61319395, 6.63516058,
                6.65665992, 6.66478415, 6.6710301, 6.67832287, 6.6987939,
                6.69954116, 6.70655977, 6.72576878, 6.77771021, 6.77863482,
                6.79102832, 6.81049338, 6.81235249, 6.81465697, 6.83783569,
                6.84815101, 6.89710246, 6.98537525, 7.01954059, 7.02622255,
                7.04976656, 7.07571722, 7.11728241, 7.13478378, 7.13478557,
                7.16044495, 7.16456219, 7.19152888, 7.19978497, 7.22787642,
                7.24906524, 7.25812186, 7.27034077, 7.30769391, 7.31820919,
                7.35549295, 7.37285349, 7.37292834, 7.37424801, 7.3785301,
                7.4196362, 7.42932103, 7.43036261, 7.45139091, 7.47555417,
                7.50122532, 7.51360212, 7.51962212, 7.55560134, 7.58438748,
                7.62698845, 7.64682633, 7.66868854, 7.6760022, 7.69020752,
                7.7238978, 7.76340706, 7.76775711, 7.79077235, 7.79151683,
                7.79383994, 7.80542945, 7.83695238, 7.85946794, 7.88079942,
                7.96879553, 7.99422322, 7.99584892, 8.09873296, 8.17614594,
                8.17763643, 8.18175172, 8.18778704, 8.22797549, 8.23708879,
                8.28821888, 8.30281824, 8.30487238, 8.33078119, 8.33420872,
                8.34305369, 8.38206152, 8.40403832, 8.41224886, 8.43463245,
                8.44389971, 8.46044352, 8.48956655, 8.51149039, 8.51796916,
                8.53329742, 8.53599617, 8.56068013, 8.56657166, 8.59814286,
                8.61214071, 8.61498351, 8.64246675, 8.65762517, 8.66282683,
                8.67384567, 8.71396613, 8.71416081, 8.73722558, 8.73767664,
                8.74798782, 8.76129767, 8.76855011, 8.80085479, 8.86199255,
                8.89862794, 8.93913818, 8.96782975, 8.9819441, 8.98865031,
                9.00024566, 9.00610235, 9.01314955, 9.02095248, 9.03094763,
                9.03668298, 9.04652449, 9.0490157, 9.05181691, 9.0646427,
                9.1264005, 9.13361863, 9.14618518, 9.15534379, 9.16200272,
                9.16524096, 9.19437442, 9.20198553, 9.20475517, 9.28953836,
                9.32111331, 9.32181408, 9.32632133, 9.32969553, 9.4558735,
                9.45868453, 9.47407654, 9.52846898, 9.54261744, 9.55992241,
                9.58831097, 9.59403646, 9.5989721, 9.63828129, 9.66338416,
                9.67033722, 9.68634843, 9.7151767, 9.72467937, 9.76497421,
                9.77592078, 9.78303691, 9.79368995, 9.7944104, 9.80563761,
                9.82690855, 9.82845111, 9.87802691, 9.90843101, 9.91777335,
                9.97014496, 9.9763017]) * pq.s, t_stop=10.0 * pq.s)]
        expected = sp.array([[0.0, 66.05735182], [66.05735182, 0.0]])
        actual = stm.victor_purpura_dist(trains)
        assert_array_almost_equal(expected, actual)

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

    def test_allows_q_to_be_zero(self):
        q = 0.0
        a = neo.SpikeTrain(
            sp.array([1.0, 2.0, 4.1, 7.0, 7.1]) * pq.s, t_stop=8.0 * pq.s)
        b = neo.SpikeTrain(
            sp.array([1.2, 4.0, 4.3]) * pq.s, t_stop=8.0 * pq.s)
        # Pure rate code
        expected = sp.array([[0.0, 2.0], [2.0, 0.0]])
        assert_array_almost_equal(expected, stm.victor_purpura_dist([a, b], q))


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
