
try:
    import unittest2 as ut
    assert ut  # Suppress pyflakes warning about redefinition of unused ut
except ImportError:
    import unittest as ut

from builders import create_empty_spike_train
from numpy.testing import assert_array_almost_equal
import neo
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
        q = 2.0 * pq.s ** -1
        expected = 0.5 * 2.0
        self.assertAlmostEqual(expected, stm.victor_purpura_dist(a, b, q))

    def test_returns_two_for_distant_spike_pair(self):
        a = neo.SpikeTrain(sp.array([1.0]) * pq.s, t_stop=6 * pq.s)
        b = neo.SpikeTrain(sp.array([5.0]) * pq.s, t_stop=6 * pq.s)
        q = 1.0 * pq.s ** -1
        expected = 2.0
        self.assertAlmostEqual(expected, stm.victor_purpura_dist(a, b, q))

    def test_returns_correct_distance_for_two_spike_trains(self):
        q = 1.0 * pq.s ** -1
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
        self.assertTrue(sp.all(sp.absolute(expected - actual) < 1e-7))

    def test_distance_of_empty_spiketrain_and_single_spike_equals_one(self):
        a = neo.SpikeTrain(sp.array([]) * pq.s, t_stop=2.0 * pq.s)
        b = neo.SpikeTrain(sp.array([1.0]) * pq.s, t_stop=2.0 * pq.s)
        expected = sp.array([[0.0, 1.0], [1.0, 0.0]])
        actual = stm.van_rossum_dist((a, b), 3.0 * pq.s)
        self.assertTrue(sp.all(sp.absolute(expected - actual) < 1e-7))


class Test_searchsorted_pairwise(ut.TestCase):
    def assert_array_tuple_equal(self, expected, actual):
        self.assertEqual(len(expected), len(actual))
        for e, a in zip(expected, actual):
            assert_array_almost_equal(
                e, a,
                err_msg="Expected: {0}\nActual: {1}".format(expected, actual))

    def test_works_with_lists(self):
        a = [1, 3, 5, 8, 9, 11]
        b = [2, 4, 6, 7, 10]
        expected = ([-1, 0, 1, 3, 3, 4], [0, 1, 2, 2, 4])
        actual = stm._searchsorted_pairwise(a, b)
        self.assert_array_tuple_equal(expected, actual)

    def test_works_with_array(self):
        a = sp.array([1, 3, 5, 8, 9, 11])
        b = sp.array([2, 4, 6, 7, 10])
        expected = ([-1, 0, 1, 3, 3, 4], [0, 1, 2, 2, 4])
        actual = stm._searchsorted_pairwise(a, b)
        self.assert_array_tuple_equal(expected, actual)

    def test_equal_items_in_second_list_are_considered_smaller(self):
        a = [1, 3, 4]
        b = [2, 3]
        expected = ([-1, 1, 1], [0, 0])
        actual = stm._searchsorted_pairwise(a, b)
        self.assert_array_tuple_equal(expected, actual)

    def test_works_with_one_list_empty(self):
        a = []
        b = [1, 2, 3]

        expected = ([], [-1, -1, -1])
        actual = stm._searchsorted_pairwise(a, b)
        self.assert_array_tuple_equal(expected, actual)

        expected = ([-1, -1, -1], [])
        actual = stm._searchsorted_pairwise(b, a)
        self.assert_array_tuple_equal(expected, actual)

    def test_returns_empty_lists_if_both_lists_are_empty(self):
        self.assert_array_tuple_equal(
            ([], []),
            stm._searchsorted_pairwise([], []))


class Test_st_inner(ut.TestCase):
    def test_returns_zero_if_any_spike_train_is_empty(self):
        empty = create_empty_spike_train()
        non_empty = neo.SpikeTrain(sp.array([1.0]) * pq.s, t_stop=2.0 * pq.s)
        kernel = sigproc.GaussianKernel()
        self.assertAlmostEqual(0.0, stm.st_inner(empty, empty, kernel))
        self.assertAlmostEqual(0.0, stm.st_inner(empty, non_empty, kernel))
        self.assertAlmostEqual(0.0, stm.st_inner(non_empty, empty, kernel))

    def test_returns_correct_inner_spike_train_product(self):
        a = neo.SpikeTrain(
            sp.array([1.0]) * pq.s, t_start=0.6 * pq.s, t_stop=1.4 * pq.s)
        b = neo.SpikeTrain(
            sp.array([0.5, 1.5]) * pq.s, t_stop=2.0 * pq.s)
        kernel = sigproc.GaussianKernel(1.0 * pq.s)
        expected = 0.530007 * pq.Hz
        actual = stm.st_inner(a, b, kernel, sampling_rate=100 * pq.Hz)
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
        k = sigproc.GaussianKernel()
        sampling_rate = 100 * pq.Hz
        self.assertAlmostEqual(
            stm.st_inner(a, b, k, sampling_rate=sampling_rate),
            stm.st_inner(b, a, k, sampling_rate=sampling_rate))


class Test_st_norm(ut.TestCase):
    def test_returns_zero_if_spike_train_is_empty(self):
        empty = create_empty_spike_train()
        kernel = sigproc.GaussianKernel()
        self.assertAlmostEqual(0.0, stm.st_norm(empty, kernel))

    def test_returns_correct_spike_train_norm(self):
        st = neo.SpikeTrain(
            sp.array([0.5, 1.0, 1.5]) * pq.s, t_stop=2.0 * pq.s)
        kernel = sigproc.GaussianKernel(1.0 * pq.s)
        expected = (2.34569 * pq.Hz) ** 0.5
        actual = stm.st_norm(st, kernel, sampling_rate=200 * pq.Hz)
        self.assertAlmostEqual(
            expected, actual.rescale(expected.units), places=3)


class Test_st_norm_dist(ut.TestCase):
    def test_returns_zero_for_equal_spike_trains(self):
        a = neo.SpikeTrain(sp.array([
            1.1844519,  1.57346687,  2.52261998,  3.65824785,  5.38988771,
            5.63178278,  6.70500182,  7.99562401,  9.21135176
        ]) * pq.s, t_stop=10.0 * pq.s, sampling_rate=100 * pq.Hz)
        k = sigproc.GaussianKernel()
        self.assertAlmostEqual(
            0.0 * pq.Hz ** 0.5, stm.st_norm_dist(a, a.copy(), k))

    def test_returns_norm_if_one_spike_train_is_empty(self):
        empty = create_empty_spike_train()
        non_empty = neo.SpikeTrain(sp.array([1.0]) * pq.s, t_stop=2.0 * pq.s)
        sampling_rate = 100 * pq.Hz
        kernel = sigproc.GaussianKernel()
        expected = stm.st_norm(non_empty, kernel, sampling_rate=sampling_rate)
        self.assertAlmostEqual(
            expected, stm.st_norm_dist(
                empty, non_empty, kernel, sampling_rate=sampling_rate),
            places=3)
        self.assertAlmostEqual(
            expected, stm.st_norm_dist(
                non_empty, empty, kernel, sampling_rate=sampling_rate),
            places=3)

    def test_returns_correct_spike_train_norm_distance(self):
        a = neo.SpikeTrain(
            sp.array([1.0]) * pq.s, t_start=0.6 * pq.s, t_stop=1.4 * pq.s)
        b = neo.SpikeTrain(
            sp.array([0.5, 1.5]) * pq.s, t_stop=2.0 * pq.s)
        kernel = sigproc.GaussianKernel(1.0 * pq.s)
        expected = (0.225662 * pq.Hz) ** 0.5
        actual = stm.st_norm_dist(a, b, kernel, sampling_rate=200 * pq.Hz)
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
        k = sigproc.GaussianKernel()
        sampling_rate = 350 * pq.Hz
        self.assertAlmostEqual(
            stm.st_norm_dist(a, b, k, sampling_rate=sampling_rate),
            stm.st_norm_dist(b, a, k, sampling_rate=sampling_rate), places=3)


if __name__ == '__main__':
    ut.main()
