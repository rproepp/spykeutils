
try:
    import unittest2 as ut
    assert ut  # Suppress pyflakes warning about redefinition of unused ut
except ImportError:
    import unittest as ut

from builders import create_empty_spike_train
from numpy.testing import assert_array_almost_equal, assert_array_equal
import neo
import quantities as pq
import scipy as sp
import spykeutils.signal_processing as sigproc


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
        actual = sigproc._searchsorted_pairwise(a, b)
        self.assert_array_tuple_equal(expected, actual)

    def test_works_with_array(self):
        a = sp.array([1, 3, 5, 8, 9, 11])
        b = sp.array([2, 4, 6, 7, 10])
        expected = ([-1, 0, 1, 3, 3, 4], [0, 1, 2, 2, 4])
        actual = sigproc._searchsorted_pairwise(a, b)
        self.assert_array_tuple_equal(expected, actual)

    def test_equal_items_in_second_list_are_considered_smaller(self):
        a = [1, 3, 4]
        b = [2, 3]
        expected = ([-1, 1, 1], [0, 0])
        actual = sigproc._searchsorted_pairwise(a, b)
        self.assert_array_tuple_equal(expected, actual)

    def test_works_with_one_list_empty(self):
        a = []
        b = [1, 2, 3]

        expected = ([], [-1, -1, -1])
        actual = sigproc._searchsorted_pairwise(a, b)
        self.assert_array_tuple_equal(expected, actual)

        expected = ([-1, -1, -1], [])
        actual = sigproc._searchsorted_pairwise(b, a)
        self.assert_array_tuple_equal(expected, actual)

    def test_returns_empty_lists_if_both_lists_are_empty(self):
        self.assert_array_tuple_equal(
            ([], []),
            sigproc._searchsorted_pairwise([], []))


class TestCausalDecayingExpKernel(ut.TestCase):
    def setUp(self):
        self.kernel_size = 500 * pq.ms
        self.kernel = sigproc.CausalDecayingExpKernel(self.kernel_size)

    def test_evaluates_to_correct_values(self):
        t = sp.array([-0.1, 0, 0.6, 1]) * pq.s
        expected = sp.array([0.0, 2.0, 0.60238842, 0.27067057]) / pq.s
        actual = self.kernel(t)
        assert_array_almost_equal(expected, actual.rescale(expected.units))

    def test_boundary_enclosing_at_least_is_correct(self):
        actual = self.kernel.boundary_enclosing_at_least(0.99)
        self.assertAlmostEqual(actual.rescale(pq.s), 2.30258509 * pq.s)


class TestGaussianKernel(ut.TestCase):
    def setUp(self):
        self.kernel_size = 500 * pq.ms
        self.kernel = sigproc.GaussianKernel(self.kernel_size)

    def test_evaluates_to_correct_values(self):
        t = sp.array([-0.1, 0, 0.6, 1]) * pq.s
        expected = sp.array([0.78208539, 0.79788456, 0.38837211, 0.10798193]) /\
            pq.s
        actual = self.kernel(t)
        assert_array_almost_equal(expected, actual.rescale(expected.units))

    def test_boundary_enclosing_at_least_is_correct(self):
        actual = self.kernel.boundary_enclosing_at_least(0.99)
        self.assertAlmostEqual(actual.rescale(pq.s), 1.28791465 * pq.s)


class TestLaplacianKernel(ut.TestCase):
    def setUp(self):
        self.kernel_size = 500 * pq.ms
        self.kernel = sigproc.LaplacianKernel(self.kernel_size)

    def test_evaluates_to_correct_values(self):
        t = sp.array([-0.1, 0, 0.6, 1]) * pq.s
        expected = sp.array([0.81873075, 1.0, 0.30119421, 0.13533528]) /\
            pq.s
        actual = self.kernel(t)
        assert_array_almost_equal(expected, actual.rescale(expected.units))

    def test_boundary_enclosing_at_least_is_correct(self):
        actual = self.kernel.boundary_enclosing_at_least(0.99)
        self.assertAlmostEqual(actual.rescale(pq.s), 2.30258509 * pq.s)


class TestRectangularKernel(ut.TestCase):
    def setUp(self):
        self.kernel_size = 500 * pq.ms
        self.kernel = sigproc.RectangularKernel(self.kernel_size)

    def test_evaluates_to_correct_values(self):
        t = sp.array([-0.1, 0, 0.6, 1]) * pq.s
        expected = sp.array([1.0, 1.0, 0.0, 0.0]) / pq.s
        actual = self.kernel(t)
        assert_array_almost_equal(expected, actual.rescale(expected.units))

    def test_boundary_enclosing_at_least_is_correct(self):
        actual = self.kernel.boundary_enclosing_at_least(0.99)
        self.assertAlmostEqual(
            actual.rescale(self.kernel_size.units), self.kernel_size)


class Test_bin_spike_train(ut.TestCase):
    def test_bins_spike_train_using_its_properties(self):
        a = neo.SpikeTrain(
            sp.array([1.0]) * pq.s, t_start=0.5 * pq.s, t_stop=1.5 * pq.s)
        a.sampling_rate = 4.0 * pq.Hz
        expected = sp.array([0, 0, 1, 0])
        expectedBins = sp.array([0.5, 0.75, 1.0, 1.25, 1.5]) * pq.s
        actual, actualBins = sigproc.bin_spike_train(a)
        assert_array_equal(expected, actual)
        assert_array_almost_equal(
            expectedBins, actualBins.rescale(expectedBins.units))

    def test_bins_spike_train_using_passed_properties(self):
        a = neo.SpikeTrain(
            sp.array([1.0]) * pq.s, t_start=0.0 * pq.s, t_stop=5.0 * pq.s)
        sampling_rate = 4.0 * pq.Hz
        t_start = 0.5 * pq.s
        t_stop = 1.5 * pq.s
        expected = sp.array([0, 0, 1, 0])
        expectedBins = sp.array([0.5, 0.75, 1.0, 1.25, 1.5]) * pq.s
        actual, actualBins = sigproc.bin_spike_train(
            a, sampling_rate=sampling_rate, t_start=t_start, t_stop=t_stop)
        assert_array_equal(expected, actual)
        assert_array_almost_equal(
            expectedBins, actualBins.rescale(expectedBins.units))


class Test_st_convolve(ut.TestCase):
    def test_convolution_with_empty_spike_train_returns_array_of_zeros(self):
        st = create_empty_spike_train()
        result, _ = sigproc.st_convolve(st, sigproc.GaussianKernel())
        self.assertTrue(sp.all(result == 0.0))

    def test_length_of_returned_array_equals_sampling_rate_times_duration(self):
        start = 2.0 * pq.s
        stop = 5.0 * pq.s
        duration = stop - start
        sampling_rate = 12 * pq.Hz
        expected_length = (sampling_rate * duration).simplified

        st = create_empty_spike_train(start, stop)
        result, _ = sigproc.st_convolve(
            st, sigproc.GaussianKernel(), sampling_rate=sampling_rate)
        self.assertEqual(expected_length, result.size)

    def test_returns_convolved_spike_train(self):
        st = neo.SpikeTrain(sp.array([1.0, 2.0]) * pq.s, t_stop=3.0 * pq.s)
        kernel = sigproc.RectangularKernel(0.3 * pq.s)
        # Because of the low sampling rate the expected result is a bit off
        # from the analytical result.
        expected = sp.array(
            [0.0, 0.0, 0.0, 1.6666666, 1.6666666, 1.6666666, 0.0,
             1.6666666, 1.6666666, 1.6666666, 0.0, 0.0]) * pq.Hz
        actual, _ = sigproc.st_convolve(st, kernel, sampling_rate=4 * pq.Hz)
        assert_array_almost_equal(expected, actual)

    def test_uses_sampling_rate_of_spike_train_if_none_is_passed(self):
        start = 2.0 * pq.s
        stop = 5.0 * pq.s
        duration = stop - start
        sampling_rate = 12 * pq.Hz
        expected_length = (sampling_rate * duration).simplified

        st = create_empty_spike_train(start, stop)
        st.sampling_rate = sampling_rate
        result, _ = sigproc.st_convolve(st, sigproc.GaussianKernel())
        self.assertEqual(expected_length, result.size)

    def test_returns_discretization_bins(self):
        start = 2.0 * pq.s
        stop = 5.0 * pq.s
        sampling_rate = 2.0 * pq.Hz
        expected = sp.array([2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]) * pq.s

        st = create_empty_spike_train(start, stop)
        _, bins = sigproc.st_convolve(
            st, sigproc.GaussianKernel(), sampling_rate=sampling_rate)
        assert_array_almost_equal(expected, bins)

    def test_mode_allows_full_convolution(self):
        start = 2.0 * pq.s
        stop = 5.0 * pq.s
        sampling_rate = 2.0 * pq.Hz
        kernel = sigproc.RectangularKernel(0.6 * pq.s)
        st = create_empty_spike_train(start, stop)

        expected_length = (stop - start) * sampling_rate + 2
        expected_bins = sp.array(
            [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5]) * pq.s

        binned, bins = sigproc.st_convolve(
            st, kernel, mode='full', sampling_rate=sampling_rate)
        self.assertEqual(binned.size, expected_length)
        assert_array_almost_equal(expected_bins, bins)

    def test_mode_allows_valid_convolution(self):
        start = 2.0 * pq.s
        stop = 5.0 * pq.s
        sampling_rate = 2.0 * pq.Hz
        kernel = sigproc.RectangularKernel(0.6 * pq.s)
        st = create_empty_spike_train(start, stop)

        expected_length = (stop - start) * sampling_rate - 2
        expected_bins = sp.array([2.5, 3.0, 3.5, 4.0, 4.5]) * pq.s

        binned, bins = sigproc.st_convolve(
            st, kernel, mode='valid', sampling_rate=sampling_rate)
        self.assertEqual(binned.size, expected_length)
        assert_array_almost_equal(expected_bins, bins)


if __name__ == '__main__':
    ut.main()
