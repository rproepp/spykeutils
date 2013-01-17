
try:
    import unittest2 as ut
    assert ut  # Suppress pyflakes warning about redefinition of unused ut
except ImportError:
    import unittest as ut

from builders import create_empty_spike_train
import neo
from numpy.testing import assert_array_almost_equal
import quantities as pq
import scipy as sp
import spykeutils.signal_processing as sigproc
import spykeutils.spike_train_metrics as stm


class CommonMetricTestCases(object):
    """ Provides some common test cases which should work for all spike train
    metrics.
    """

    def calc_metric(self, a, b):
        """ Calculates and returns the metric under test.

        :param SpikeTrain a:
        :param SpikeTrain b:
        :rtype: float
        """
        raise NotImplementedError()

    def test_is_zero_for_identical_spike_trains(self):
        st = neo.SpikeTrain(
            sp.array([1, 2, 3]) * pq.s, t_start=0 * pq.s, t_stop=4 * pq.s)
        self.assertAlmostEqual(0, self.calc_metric(st, st.copy()))

    def test_works_with_empty_spike_trains(self):
        st = neo.SpikeTrain(sp.array([]) * pq.s, t_stop=2.0 * pq.s)
        self.assertAlmostEqual(0, self.calc_metric(st, st.copy()))

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
        self.assertAlmostEqual(self.calc_metric(a, b), self.calc_metric(b, a))


class Test_victor_purpura_dist(ut.TestCase, CommonMetricTestCases):
    def calc_metric(self, a, b):
        return stm.victor_purpura_dist(a, b)

    def test_inserted_spikes_equal_cost_of_one(self):
        num_spikes = 3
        st = neo.SpikeTrain(sp.arange(3) * pq.s, t_stop=num_spikes * pq.s)
        st_empty = create_empty_spike_train()
        self.assertAlmostEqual(
            num_spikes, stm.victor_purpura_dist(st, st_empty))

    def test_deleted_spikes_equal_cost_of_one(self):
        num_spikes = 3
        st = neo.SpikeTrain(sp.arange(3) * pq.s, t_stop=num_spikes * pq.s)
        st_empty = create_empty_spike_train()
        self.assertAlmostEqual(
            num_spikes, stm.victor_purpura_dist(st_empty, st))

    def test_returns_q_weighted_dist_for_close_spike_pair(self):
        a = neo.SpikeTrain(sp.array([1.0]) * pq.s, t_stop=2 * pq.s)
        b = neo.SpikeTrain(sp.array([1.5]) * pq.s, t_stop=2 * pq.s)
        q = 2.0 / pq.s
        expected = 0.5 * 2.0
        self.assertAlmostEqual(expected, stm.victor_purpura_dist(a, b, q))

    def test_returns_two_for_distant_spike_pair(self):
        a = neo.SpikeTrain(sp.array([1.0]) * pq.s, t_stop=6 * pq.s)
        b = neo.SpikeTrain(sp.array([5.0]) * pq.s, t_stop=6 * pq.s)
        q = 1.0 / pq.s
        expected = 2.0
        self.assertAlmostEqual(expected, stm.victor_purpura_dist(a, b, q))

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
        expected = 4.3
        self.assertAlmostEqual(expected, stm.victor_purpura_dist(a, b, q))

    def test_allows_use_of_different_kernel(self):
        k = sigproc.LaplacianKernel(1.0 * pq.s, normalize=False)
        a = neo.SpikeTrain(
            sp.array([1.0, 2.0, 4.1, 7.0, 7.1]) * pq.s, t_stop=8.0 * pq.s)
        b = neo.SpikeTrain(
            sp.array([1.2, 4.0, 4.3]) * pq.s, t_stop=8.0 * pq.s)
        # From a to b:
        #   - shift 1.0 to 1.2 (cost 0.3625385)
        #   - delete 2.0 (cost 1.0)
        #   - shift 4.1 to 4.0 (cost 0.1903252)
        #   - shift 4.3 to 7.0 (cost 1.8655890)
        #   - delete 7.0 (cost 1.0)
        #   - delete 7.1 (cost 1.0)
        expected = 4.4184526
        self.assertAlmostEqual(
            expected, stm.victor_purpura_dist(a, b, kernel=k))


class Test_victor_purpura_multiunit_dist(ut.TestCase, CommonMetricTestCases):
    # With only one spike train each we should get the normal VP distance.
    def calc_metric(self, a, b):
        return stm.victor_purpura_multiunit_dist({0: a}, {0: b}, 1)

    def test_returns_correct_distance_for_multiunits(self):
        a0 = neo.SpikeTrain(sp.array([1.0, 5.0, 7.0]) * pq.s, t_stop=8.0 * pq.s)
        a1 = neo.SpikeTrain(sp.array([2.0, 4.0, 5.0]) * pq.s, t_stop=8.0 * pq.s)
        b0 = neo.SpikeTrain(sp.array([1.0, 2.0, 5.0]) * pq.s, t_stop=8.0 * pq.s)
        b1 = neo.SpikeTrain(sp.array([3.0, 8.0]) * pq.s, t_stop=9.0 * pq.s)
        a = {0: a0, 1: a1}
        b = {1: b1, 0: b0}
        reassignment_cost = 0.7
        expected = 4.4
        actual = stm.victor_purpura_multiunit_dist(a, b, reassignment_cost)
        self.assertAlmostEqual(expected, actual)
        actual = stm.victor_purpura_multiunit_dist(b, a, reassignment_cost)
        self.assertAlmostEqual(expected, actual)


class Test_van_rossum_dist(ut.TestCase, CommonMetricTestCases):
    def calc_metric(self, a, b):
        return stm.van_rossum_dist((a, b))[0, 1]

    def test_return_correct_distance(self):
        a = neo.SpikeTrain(
            sp.array([1.0, 4.0, 5.0, 6.0, 9.0, 11.0]) * pq.s,
            t_stop=12.0 * pq.s)
        b = neo.SpikeTrain(
            sp.array([2.0, 4.0, 7.0, 10.0]) * pq.s,  t_stop=12.0 * pq.s)
        c = neo.SpikeTrain(sp.array([3.0, 4.0]) * pq.s, t_stop=12.0 * pq.s)
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
    def calc_metric(self, a, b):
        return stm.van_rossum_multiunit_dist({0: a}, {0: b}, 1)

    def test_returns_correct_distance_for_multiunits(self):
        a0 = neo.SpikeTrain(sp.array([1.0, 5.0, 7.0]) * pq.s, t_stop=8.0 * pq.s)
        a1 = neo.SpikeTrain(sp.array([2.0, 4.0, 5.0]) * pq.s, t_stop=8.0 * pq.s)
        b0 = neo.SpikeTrain(sp.array([1.0, 2.0, 5.0]) * pq.s, t_stop=8.0 * pq.s)
        b1 = neo.SpikeTrain(sp.array([3.0, 8.0]) * pq.s, t_stop=9.0 * pq.s)
        a = {0: a0, 1: a1}
        b = {1: b1, 0: b0}
        weighting = 0.3
        expected = 2.37006181
        actual = stm.van_rossum_multiunit_dist(a, b, weighting)
        self.assertAlmostEqual(expected, actual)
        actual = stm.van_rossum_multiunit_dist(b, a, weighting)
        self.assertAlmostEqual(expected, actual)

    def test_allows_tau_equal_to_infinity_with_multiunits(self):
        a0 = neo.SpikeTrain(sp.array([1.0, 5.0, 7.0]) * pq.s, t_stop=8.0 * pq.s)
        a1 = neo.SpikeTrain(sp.array([2.0, 4.0, 5.0]) * pq.s, t_stop=8.0 * pq.s)
        b0 = neo.SpikeTrain(sp.array([5.0]) * pq.s, t_stop=8.0 * pq.s)
        b1 = neo.SpikeTrain(sp.array([3.0, 8.0]) * pq.s, t_stop=9.0 * pq.s)
        a = {0: a0, 1: a1}
        b = {1: b1, 0: b0}
        weighting = 0.3
        tau = sp.inf * pq.s
        expected = sp.sqrt(5.0 + weighting * 4.0)
        actual = stm.van_rossum_multiunit_dist(a, b, weighting, tau)
        assert_array_almost_equal(expected, actual)


#class Test_st_inner(ut.TestCase):
    #def test_returns_zero_if_any_spike_train_is_empty(self):
        #empty = create_empty_spike_train()
        #non_empty = neo.SpikeTrain(sp.array([1.0]) * pq.s, t_stop=2.0 * pq.s)
        #smoothing_filter = sigproc.GaussianKernel()
        #self.assertAlmostEqual(
            #0.0, stm.st_inner(empty, empty, smoothing_filter))
        #self.assertAlmostEqual(
            #0.0, stm.st_inner(empty, non_empty, smoothing_filter))
        #self.assertAlmostEqual(
            #0.0, stm.st_inner(non_empty, empty, smoothing_filter))

    #def test_returns_correct_inner_spike_train_product(self):
        #a = neo.SpikeTrain(
            #sp.array([1.0]) * pq.s, t_start=0.6 * pq.s, t_stop=1.4 * pq.s)
        #b = neo.SpikeTrain(
            #sp.array([0.5, 1.5]) * pq.s, t_stop=2.0 * pq.s)
        #smoothing_filter = sigproc.GaussianKernel(1.0 * pq.s)
        #expected = 0.530007 * pq.Hz
        #actual = stm.st_inner(a, b, smoothing_filter, sampling_rate=100 * pq.Hz)
        #self.assertAlmostEqual(
            #expected, actual.rescale(expected.units), places=3)

    #def test_is_symmetric(self):
        #a = neo.SpikeTrain(sp.array([
            #1.1844519,  1.57346687,  2.52261998,  3.65824785,  5.38988771,
            #5.63178278,  6.70500182,  7.99562401,  9.21135176
        #]) * pq.s, t_stop=10.0 * pq.s)
        #b = neo.SpikeTrain(sp.array([
            #0.86096077,  3.54273148,  4.20476326,  6.02451599,  6.42851683,
            #6.5564268,  7.07864592,  7.2368936,  7.31784319,  8.15148958,
            #8.53540889
        #]) * pq.s, t_stop=10.0 * pq.s)
        #f = sigproc.GaussianKernel()
        #sampling_rate = 100 * pq.Hz
        #self.assertAlmostEqual(
            #stm.st_inner(a, b, f, sampling_rate=sampling_rate),
            #stm.st_inner(b, a, f, sampling_rate=sampling_rate))


#class Test_st_norm(ut.TestCase):
    #def test_returns_zero_if_spike_train_is_empty(self):
        #empty = create_empty_spike_train()
        #smoothing_filter = sigproc.GaussianKernel()
        #self.assertAlmostEqual(0.0, stm.st_norm(empty, smoothing_filter))

    #def test_returns_correct_spike_train_norm(self):
        #st = neo.SpikeTrain(
            #sp.array([0.5, 1.0, 1.5]) * pq.s, t_stop=2.0 * pq.s)
        #smoothing_filter = sigproc.GaussianKernel(1.0 * pq.s)
        #expected = (2.34569 * pq.Hz) ** 0.5
        #actual = stm.st_norm(st, smoothing_filter, sampling_rate=200 * pq.Hz)
        #self.assertAlmostEqual(
            #expected, actual.rescale(expected.units), places=3)


#class Test_norm_dist(ut.TestCase):
    #def test_returns_zero_for_equal_spike_trains(self):
        #a = neo.SpikeTrain(sp.array([
            #1.1844519,  1.57346687,  2.52261998,  3.65824785,  5.38988771,
            #5.63178278,  6.70500182,  7.99562401,  9.21135176
        #]) * pq.s, t_stop=10.0 * pq.s, sampling_rate=100 * pq.Hz)
        #f = sigproc.GaussianKernel()
        #self.assertAlmostEqual(
            #0.0 * pq.Hz ** 0.5, stm.norm_dist(a, a.copy(), f))

    #def test_returns_norm_if_one_spike_train_is_empty(self):
        #empty = create_empty_spike_train()
        #non_empty = neo.SpikeTrain(sp.array([1.0]) * pq.s, t_stop=2.0 * pq.s)
        #sampling_rate = 100 * pq.Hz
        #smoothing_filter = sigproc.GaussianKernel()
        #expected = stm.st_norm(
            #non_empty, smoothing_filter, sampling_rate=sampling_rate)
        #self.assertAlmostEqual(
            #expected, stm.norm_dist(
                #empty, non_empty, smoothing_filter,
                #sampling_rate=sampling_rate),
            #places=3)
        #self.assertAlmostEqual(
            #expected, stm.norm_dist(
                #non_empty, empty, smoothing_filter,
                #sampling_rate=sampling_rate),
            #places=3)

    #def test_returns_correct_spike_train_norm_distance(self):
        #a = neo.SpikeTrain(
            #sp.array([1.0]) * pq.s, t_start=0.6 * pq.s, t_stop=1.4 * pq.s)
        #b = neo.SpikeTrain(
            #sp.array([0.5, 1.5]) * pq.s, t_stop=2.0 * pq.s)
        #smoothing_filter = sigproc.GaussianKernel(1.0 * pq.s)
        #expected = (0.225662 * pq.Hz) ** 0.5
        #actual = stm.norm_dist(
            #a, b, smoothing_filter, sampling_rate=200 * pq.Hz)
        #self.assertAlmostEqual(
            #expected, actual.rescale(expected.units), places=3)

    #def test_is_symmetric(self):
        #a = neo.SpikeTrain(sp.array([
            #1.1844519,  1.57346687,  2.52261998,  3.65824785,  5.38988771,
            #5.63178278,  6.70500182,  7.99562401,  9.21135176
        #]) * pq.s, t_stop=10.0 * pq.s)
        #b = neo.SpikeTrain(sp.array([
            #0.86096077,  3.54273148,  4.20476326,  6.02451599,  6.42851683,
            #6.5564268,  7.07864592,  7.2368936,  7.31784319,  8.15148958,
            #8.53540889
        #]) * pq.s, t_stop=10.0 * pq.s)
        #f = sigproc.GaussianKernel()
        #sampling_rate = 350 * pq.Hz
        #self.assertAlmostEqual(
            #stm.norm_dist(a, b, f, sampling_rate=sampling_rate),
            #stm.norm_dist(b, a, f, sampling_rate=sampling_rate), places=3)


#class Test_cs_dist(ut.TestCase):
    #def test_returns_zero_for_equal_spike_trains(self):
        #a = neo.SpikeTrain(sp.array([
            #1.1844519,  1.57346687,  2.52261998,  3.65824785,  5.38988771,
            #5.63178278,  6.70500182,  7.99562401,  9.21135176
        #]) * pq.s, t_stop=10.0 * pq.s, sampling_rate=100 * pq.Hz)
        #f = sigproc.GaussianKernel()
        #self.assertAlmostEqual(0.0, stm.cs_dist(a, a.copy(), f))

    #def test_returns_nan_if_one_spike_train_is_empty(self):
        #empty = create_empty_spike_train()
        #non_empty = neo.SpikeTrain(sp.array([1.0]) * pq.s, t_stop=2.0 * pq.s)
        #sampling_rate = 100 * pq.Hz
        #smoothing_filter = sigproc.GaussianKernel()
        #self.assertTrue(sp.isnan(stm.cs_dist(
            #empty, empty, smoothing_filter,
            #sampling_rate=sampling_rate)))
        #self.assertTrue(sp.isnan(stm.cs_dist(
            #empty, non_empty, smoothing_filter,
            #sampling_rate=sampling_rate)))
        #self.assertTrue(sp.isnan(stm.cs_dist(
            #non_empty, empty, smoothing_filter,
            #sampling_rate=sampling_rate)))

    #def test_returns_correct_spike_train_cauchy_schwarz_distance(self):
        #a = neo.SpikeTrain(
            #sp.array([1.0]) * pq.s, t_start=0.6 * pq.s, t_stop=1.4 * pq.s)
        #b = neo.SpikeTrain(
            #sp.array([0.5, 1.5]) * pq.s, t_stop=2.0 * pq.s)
        #smoothing_filter = sigproc.GaussianKernel(1.0 * pq.s)
        #expected = 0.124677
        #actual = stm.cs_dist(a, b, smoothing_filter, sampling_rate=200 * pq.Hz)
        #self.assertAlmostEqual(expected, actual, places=3)

    #def test_is_symmetric(self):
        #a = neo.SpikeTrain(sp.array([
            #1.1844519,  1.57346687,  2.52261998,  3.65824785,  5.38988771,
            #5.63178278,  6.70500182,  7.99562401,  9.21135176
        #]) * pq.s, t_stop=10.0 * pq.s)
        #b = neo.SpikeTrain(sp.array([
            #0.86096077,  3.54273148,  4.20476326,  6.02451599,  6.42851683,
            #6.5564268,  7.07864592,  7.2368936,  7.31784319,  8.15148958,
            #8.53540889
        #]) * pq.s, t_stop=10.0 * pq.s)
        #f = sigproc.GaussianKernel()
        #sampling_rate = 350 * pq.Hz
        #self.assertAlmostEqual(
            #stm.cs_dist(a, b, f, sampling_rate=sampling_rate),
            #stm.cs_dist(b, a, f, sampling_rate=sampling_rate), places=3)


#class Test_schreiber_similarity(ut.TestCase):
    #def test_returns_one_for_equal_spike_trains(self):
        #a = neo.SpikeTrain(sp.array([
            #1.1844519,  1.57346687,  2.52261998,  3.65824785,  5.38988771,
            #5.63178278,  6.70500182,  7.99562401,  9.21135176
        #]) * pq.s, t_stop=10.0 * pq.s)
        #k = sigproc.GaussianKernel()
        #actual = stm.schreiber_similarity((a, a.copy()), k)
        #self.assertAlmostEqual(1.0, actual[0, 1])

    #def test_returns_nan_if_one_spike_train_is_empty(self):
        #empty = create_empty_spike_train()
        #non_empty = neo.SpikeTrain(sp.array([1.0]) * pq.s, t_stop=2.0 * pq.s)
        #k = sigproc.GaussianKernel()
        #actual = stm.schreiber_similarity((empty, non_empty), k)
        #self.assertTrue(sp.isnan(actual[0, 0]))
        #self.assertTrue(sp.isnan(actual[0, 1]))
        #self.assertTrue(sp.isnan(actual[1, 0]))

    #def test_returns_correct_spike_train_schreiber_similarity(self):
        #a = neo.SpikeTrain(
            #sp.array([1.0]) * pq.s, t_start=0.6 * pq.s, t_stop=1.4 * pq.s)
        #b = neo.SpikeTrain(
            #sp.array([0.5, 1.5]) * pq.s, t_stop=2.0 * pq.s)
        #c = neo.SpikeTrain(
            #sp.array([1.0, 2.0]) * pq.s, t_start=0.6 * pq.s, t_stop=2.4 * pq.s)
        #k = sigproc.GaussianKernel(sp.sqrt(2.0) * pq.s)
        #expected = sp.array([
            #[1.0, 0.9961114, 0.9430803],
            #[0.9961114, 1.0, 0.9523332],
            #[0.9430803, 0.9523332, 1.0]])
        #actual = stm.schreiber_similarity((a, b, c), k)
        #assert_array_almost_equal(expected, actual)

    #def test_is_symmetric(self):
        #a = neo.SpikeTrain(sp.array([
            #1.1844519,  1.57346687,  2.52261998,  3.65824785,  5.38988771,
            #5.63178278,  6.70500182,  7.99562401,  9.21135176
        #]) * pq.s, t_stop=10.0 * pq.s)
        #b = neo.SpikeTrain(sp.array([
            #0.86096077,  3.54273148,  4.20476326,  6.02451599,  6.42851683,
            #6.5564268,  7.07864592,  7.2368936,  7.31784319,  8.15148958,
            #8.53540889
        #]) * pq.s, t_stop=10.0 * pq.s)
        #k = sigproc.GaussianKernel()
        #assert_array_almost_equal(
            #stm.schreiber_similarity((a, b), k),
            #stm.schreiber_similarity((b, a), k))


if __name__ == '__main__':
    ut.main()
