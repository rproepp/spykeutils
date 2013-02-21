try:
    import unittest2 as ut
    assert ut  # Suppress pyflakes warning about redefinition of unused ut
except ImportError:
    import unittest as ut

from builders import arange_spikes
from numpy.testing import assert_array_equal, assert_array_almost_equal
import spykeutils.rate_estimation as re
import neo
import quantities as pq
import scipy as sp


class Test_binned_spike_trains(ut.TestCase):
    def test_bins_spike_train_using_its_properties(self):
        a = neo.SpikeTrain(
            sp.array([1.0]) * pq.s, t_start=0.5 * pq.s, t_stop=1.5 * pq.s)
        bin_size = 250.0 * pq.ms
        expected = {0: [sp.array([0, 0, 1])]}
        expectedBins = sp.array([0.5, 0.75, 1.0, 1.25]) * pq.s
        actual, actualBins = re.binned_spike_trains({0: [a]}, bin_size)
        self.assertEqual(len(expected), len(actual))
        self.assertEqual(len(expected[0]), len(actual[0]))
        assert_array_equal(expected[0][0], actual[0][0])
        assert_array_almost_equal(
            expectedBins, actualBins.rescale(expectedBins.units))

    def test_bins_spike_train_using_passed_properties(self):
        a = neo.SpikeTrain(
            sp.array([1.0]) * pq.s, t_start=0.0 * pq.s, t_stop=5.0 * pq.s)
        bin_size = 250.0 * pq.ms
        t_start = 0.5 * pq.s
        t_stop = 1.5 * pq.s
        expected = {0: [sp.array([0, 0, 1])]}
        expectedBins = sp.array([0.5, 0.75, 1.0, 1.25]) * pq.s
        actual, actualBins = re.binned_spike_trains(
            {0: [a]}, bin_size, start=t_start, stop=t_stop)
        self.assertEqual(len(expected), len(actual))
        self.assertEqual(len(expected[0]), len(actual[0]))
        assert_array_equal(expected[0][0], actual[0][0])
        assert_array_almost_equal(
            expectedBins, actualBins.rescale(expectedBins.units))

    def test_uses_min_spike_train_interval(self):
        a = arange_spikes(5 * pq.s)
        b = arange_spikes(3 * pq.s, 15 * pq.s)
        bin_size = 250.0 * pq.ms
        expectedBins = sp.arange(3.0, 4.9, 0.25) * pq.s
        actual, actualBins = re.binned_spike_trains({0: [a, b]}, bin_size)
        assert_array_almost_equal(
            expectedBins, actualBins.rescale(expectedBins.units))

    def test_handles_bin_size_which_is_not_divisor_of_duration(self):
        a = arange_spikes(5 * pq.s)
        bin_size = 1300.0 * pq.ms
        expected = {0: [sp.array([1, 1, 1])]}
        expectedBins = sp.array([0.0, 1.3, 2.6, 3.9]) * pq.s
        actual, actualBins = re.binned_spike_trains({0: [a]}, bin_size)
        self.assertEqual(len(expected), len(actual))
        self.assertEqual(len(expected[0]), len(actual[0]))
        assert_array_equal(expected[0][0], actual[0][0])
        assert_array_almost_equal(
            expectedBins, actualBins.rescale(expectedBins.units))


if __name__ == '__main__':
    ut.main()
