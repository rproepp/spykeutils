
try:
    import unittest2 as ut
    assert ut  # Suppress pyflakes warning about redefinition of unused ut
except ImportError:
    import unittest as ut

from builders import create_empty_spike_train
from mock import MagicMock
from numpy.testing import assert_array_almost_equal, assert_array_equal
import neo
import quantities as pq
import scipy as sp
import spykeutils.signal_processing as sigproc


class TestKernel(ut.TestCase):
    def test_call_returns_result_of_evaluate(self):
        kernel_size = 1.3
        kernel = sigproc.Kernel(kernel_size, normalize=False)
        kernel._evaluate = MagicMock(name='_evaluate')
        kernel._evaluate.return_value = 42

        t = 3.3
        actual = kernel(t)

        kernel._evaluate.assert_called_with(t, kernel_size)
        self.assertEqual(kernel._evaluate.return_value, actual)

    def test_call_can_normalize_evaluate(self):
        kernel_size = 1.3
        kernel = sigproc.Kernel(kernel_size, normalize=True)
        kernel._evaluate = MagicMock(name='_evaluate')
        kernel._evaluate.return_value = 42
        kernel.normalization_factor = MagicMock(name='normalization_factor')
        kernel.normalization_factor.return_value = 0.5

        t = 3.3
        actual = kernel(t)

        kernel._evaluate.assert_called_with(t, kernel_size)
        kernel.normalization_factor.assert_called_with(kernel_size)
        self.assertEqual(
            kernel.normalization_factor.return_value *
            kernel._evaluate.return_value, actual)

    def test_call_can_overwrite_kernel_size(self):
        kernel_size = 1.3
        kernel = sigproc.Kernel(5.6, normalize=True)
        kernel._evaluate = MagicMock(name='_evaluate')
        kernel._evaluate.return_value = 42
        kernel.normalization_factor = MagicMock(name='normalization_factor')
        kernel.normalization_factor.return_value = 0.5

        t = 3.3
        actual = kernel(t, kernel_size)

        kernel._evaluate.assert_called_with(t, kernel_size)
        kernel.normalization_factor.assert_called_with(kernel_size)
        self.assertEqual(
            kernel.normalization_factor.return_value *
            kernel._evaluate.return_value, actual)

    def test_summed_dist_matrix(self):
        kernel = sigproc.Kernel(1.0, normalize=False)
        kernel._evaluate = lambda t, _: t
        vectors = [sp.array([2.0, 1.0, 3.0]), sp.array([1.5, 4.0])]
        expected = sp.array([[0.0, -4.5], [4.5, 0.0]])
        actual = kernel.summed_dist_matrix(vectors)
        assert_array_almost_equal(expected, actual)

    def test_summed_dist_matrix_with_units(self):
        kernel = sigproc.Kernel(2000 * pq.ms, normalize=False)
        kernel._evaluate = lambda t, size: 1.0 / size / size * t
        vectors = [sp.array([2.0, 1.0, 3.0]) * pq.s,
                   sp.array([1500, 4000]) * pq.ms]
        expected = sp.array([[0.0, -1.125], [1.125, 0.0]]) / pq.s
        actual = kernel.summed_dist_matrix(vectors)
        assert_array_almost_equal(expected, actual.rescale(1.0 / pq.s))


class TestSymmetricKernel(ut.TestCase):
    def test_summed_dist_matrix(self):
        kernel = sigproc.Kernel(1.0, normalize=False)
        kernel._evaluate = lambda t, _: sp.absolute(t)
        vectors = [sp.array([2.0, 1.0, 3.0]), sp.array([1.5, 4.0])]
        expected = sp.array([[8.0, 8.5], [8.5, 5.0]])
        actual = kernel.summed_dist_matrix(vectors)
        assert_array_almost_equal(expected, actual)

    def test_summed_dist_matrix_with_units(self):
        kernel = sigproc.Kernel(2000 * pq.ms, normalize=False)
        kernel._evaluate = lambda t, size: 1.0 / size / size * sp.absolute(t)
        vectors = [sp.array([2.0, 1.0, 3.0]) * pq.s,
                   sp.array([1500, 4000]) * pq.ms]
        expected = sp.array([[2.0, 2.125], [2.125, 1.25]]) / pq.s
        actual = kernel.summed_dist_matrix(vectors)
        assert_array_almost_equal(expected, actual.rescale(1.0 / pq.s))


class Test_discretize_kernel(ut.TestCase):
    def test_discretizes_requested_area(self):
        kernel = sigproc.Kernel(1.0, normalize=False)
        kernel.boundary_enclosing_at_least = MagicMock(
            name='boundary_enclosing_at_least')
        kernel.boundary_enclosing_at_least.return_value = 2.0
        kernel._evaluate = lambda x, _: sp.ones(len(x))
        sampling_rate = 1.0
        mock_discretization = sp.ones(5)

        kernel_area_fraction = 0.5
        actual = sigproc.discretize_kernel(
            kernel, sampling_rate, kernel_area_fraction)

        kernel.boundary_enclosing_at_least.assert_called_with(
            kernel_area_fraction)
        assert_array_equal(actual, mock_discretization)

    def test_discretizes_requested_area_with_units(self):
        kernel = sigproc.Kernel(100.0 * pq.ms, normalize=False)
        kernel.boundary_enclosing_at_least = MagicMock(
            name='boundary_enclosing_at_least')
        kernel.boundary_enclosing_at_least.return_value = 100.0 * pq.ms
        kernel._evaluate = lambda x, _: sp.ones(len(x))
        sampling_rate = 10.0 * pq.Hz
        mock_discretization = sp.ones(3)

        kernel_area_fraction = 0.5
        actual = sigproc.discretize_kernel(
            kernel, sampling_rate, kernel_area_fraction)

        kernel.boundary_enclosing_at_least.assert_called_with(
            kernel_area_fraction)
        assert_array_equal(actual, mock_discretization)

    def test_discretizes_requested_number_of_bins(self):
        kernel = sigproc.Kernel(1.0, normalize=False)
        kernel._evaluate = lambda x, _: sp.ones(len(x))
        sampling_rate = 1.0
        num_bins = 23
        mock_discretization = sp.ones(num_bins)

        actual = sigproc.discretize_kernel(
            kernel, sampling_rate, num_bins=num_bins)
        assert_array_equal(actual, mock_discretization)

    def test_can_normalize_to_unit_area(self):
        kernel = sigproc.Kernel(1.0, normalize=False)
        kernel._evaluate = lambda x, _: sp.ones(len(x))
        sampling_rate = 2.0
        num_bins = 20
        mock_discretization = sp.ones(num_bins) / num_bins * sampling_rate

        actual = sigproc.discretize_kernel(
            kernel, sampling_rate, num_bins=num_bins, ensure_unit_area=True)
        assert_array_equal(actual, mock_discretization)


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

    def test_summed_dist_matrix(self):
        kernel = sigproc.LaplacianKernel(1.0, normalize=False)
        vectors = [sp.array([2.0, 1.0, 3.0]), sp.array([1.5, 4.0])]
        expected = sp.array(
            [[4.7421883311589941, 1.9891932723496157],
             [1.9891932723496157, 2.1641699972477975]])
        actual = kernel.summed_dist_matrix(vectors)
        assert_array_almost_equal(expected, actual)

    def test_summed_dist_matrix_with_units(self):
        kernel = sigproc.LaplacianKernel(1000 * pq.ms, normalize=True)
        vectors = [sp.array([2.0, 1.0, 3.0]) * pq.s,
                   sp.array([1500, 4000]) * pq.ms]
        expected = sp.array(
            [[4.7421883311589941, 1.9891932723496157],
             [1.9891932723496157, 2.1641699972477975]]) / 2.0 / pq.s
        actual = kernel.summed_dist_matrix(vectors)
        assert_array_almost_equal(expected, actual.rescale(1.0 / pq.s))


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


class TestTriangularKernel(ut.TestCase):
    def setUp(self):
        self.kernel_size = 500 * pq.ms
        self.kernel = sigproc.TriangularKernel(self.kernel_size)

    def test_evaluates_to_correct_values(self):
        t = sp.array([-0.7, -0.1, 0, 0.3, 0.6, 1]) * pq.s
        expected = 2 * sp.array([0.0, 0.8, 1.0, 0.4, 0.0, 0.0]) / pq.s
        actual = self.kernel(t)
        assert_array_almost_equal(expected, actual.rescale(expected.units))

    def test_boundary_enclosing_at_least_is_correct(self):
        actual = self.kernel.boundary_enclosing_at_least(0.99)
        self.assertAlmostEqual(
            actual.rescale(self.kernel_size.units), self.kernel_size)


class Test_smooth(ut.TestCase):
    def test_convolution_with_empty_binned_array_returns_array_of_zeros(self):
        binned = sp.zeros(10)
        sampling_rate = 10 * pq.Hz
        result = sigproc.smooth(binned, sigproc.GaussianKernel(), sampling_rate)
        self.assertTrue(sp.all(result == 0.0))

    def test_length_of_returned_array_equals_length_of_binned(self):
        binned = sp.ones(10)
        sampling_rate = 10 * pq.Hz
        result = sigproc.smooth(binned, sigproc.GaussianKernel(), sampling_rate)
        self.assertEqual(binned.size, result.size)

    def test_returns_smoothed_representation(self):
        binned = sp.array([0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0])
        sampling_rate = 4 * pq.Hz
        kernel = sigproc.RectangularKernel(0.3 * pq.s)
        # Because of the low sampling rate the expected result is a bit off
        # from the analytical result.
        expected = sp.array(
            [0.0, 0.0, 0.0, 1.6666666, 1.6666666, 1.6666666, 0.0,
             1.6666666, 1.6666666, 1.6666666, 0.0, 0.0]) * pq.Hz
        actual = sigproc.smooth(binned, kernel, sampling_rate=sampling_rate)
        assert_array_almost_equal(expected, actual)

    def test_mode_allows_full_convolution(self):
        binned = sp.ones(10)
        sampling_rate = 2.0 * pq.Hz
        kernel = sigproc.RectangularKernel(0.6 * pq.s)
        expected_length = 12
        actual = sigproc.smooth(
            binned, kernel, sampling_rate=sampling_rate, mode='full')
        self.assertEqual(actual.size, expected_length)

    def test_mode_allows_valid_convolution(self):
        binned = sp.ones(10)
        sampling_rate = 2.0 * pq.Hz
        kernel = sigproc.RectangularKernel(0.6 * pq.s)
        expected_length = 8
        actual = sigproc.smooth(
            binned, kernel, sampling_rate=sampling_rate, mode='valid')
        self.assertEqual(actual.size, expected_length)


class Test_st_convolve(ut.TestCase):
    def test_convolution_with_empty_spike_train_returns_array_of_zeros(self):
        st = create_empty_spike_train()
        result, _ = sigproc.st_convolve(st, sigproc.GaussianKernel(), 1 * pq.Hz)
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
