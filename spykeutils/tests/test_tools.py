

try:
    import unittest2 as ut
    assert ut  # Suppress pyflakes warning about redefinition of unused ut
except ImportError:
    import unittest as ut

from builders import arange_spikes
from numpy.testing import assert_array_equal, assert_array_almost_equal
from spykeutils import tools
import neo
import quantities as pq
import scipy as sp


class Test_apply_to_dict(ut.TestCase):
    @staticmethod
    def fn(train, multiplier=1):
        return multiplier * train.size

    def test_maps_function_to_each_spike_train(self):
        st_dict = {'a': [arange_spikes(5 * pq.s), arange_spikes(4 * pq.s)],
                   'b': [arange_spikes(7 * pq.s)]}
        expected = {'a': [4, 3], 'b': [6]}
        actual = tools.apply_to_dict(self.fn, st_dict)
        self.assertEqual(expected, actual)

    def test_works_on_empty_lists(self):
        st_dict = {'a': [], 'b': []}
        expected = {'a': [], 'b': []}
        actual = tools.apply_to_dict(self.fn, st_dict)
        self.assertEqual(expected, actual)

    def test_works_on_empty_dict(self):
        st_dict = {}
        expected = {}
        actual = tools.apply_to_dict(self.fn, st_dict)
        self.assertEqual(expected, actual)

    def test_allows_to_pass_additional_args(self):
        st_dict = {'a': [arange_spikes(5 * pq.s), arange_spikes(4 * pq.s)],
                   'b': [arange_spikes(7 * pq.s)]}
        expected = {'a': [8, 6], 'b': [12]}
        actual = tools.apply_to_dict(self.fn, st_dict, 2)
        self.assertEqual(expected, actual)


class Test_bin_spike_trains(ut.TestCase):
    def test_bins_spike_train_using_its_properties(self):
        a = neo.SpikeTrain(
            sp.array([1.0]) * pq.s, t_start=0.5 * pq.s, t_stop=1.5 * pq.s)
        sampling_rate = 4.0 * pq.Hz
        expected = {0: [sp.array([0, 0, 1, 0])]}
        expectedBins = sp.array([0.5, 0.75, 1.0, 1.25, 1.5]) * pq.s
        actual, actualBins = tools.bin_spike_trains({0: [a]}, sampling_rate)
        self.assertEqual(len(expected), len(actual))
        self.assertEqual(len(expected[0]), len(actual[0]))
        assert_array_equal(expected[0][0], actual[0][0])
        assert_array_almost_equal(
            expectedBins, actualBins.rescale(expectedBins.units))

    def test_bins_spike_train_using_passed_properties(self):
        a = neo.SpikeTrain(
            sp.array([1.0]) * pq.s, t_start=0.0 * pq.s, t_stop=5.0 * pq.s)
        sampling_rate = 4.0 * pq.Hz
        t_start = 0.5 * pq.s
        t_stop = 1.5 * pq.s
        expected = {0: [sp.array([0, 0, 1, 0])]}
        expectedBins = sp.array([0.5, 0.75, 1.0, 1.25, 1.5]) * pq.s
        actual, actualBins = tools.bin_spike_trains(
            {0: [a]}, sampling_rate=sampling_rate, t_start=t_start,
            t_stop=t_stop)
        self.assertEqual(len(expected), len(actual))
        self.assertEqual(len(expected[0]), len(actual[0]))
        assert_array_equal(expected[0][0], actual[0][0])
        assert_array_almost_equal(
            expectedBins, actualBins.rescale(expectedBins.units))

    def test_uses_max_spike_train_interval(self):
        a = arange_spikes(5 * pq.s)
        b = arange_spikes(7 * pq.s, 15 * pq.s)
        sampling_rate = 4.0 * pq.Hz
        expectedBins = sp.arange(0.0, 15.1, 0.25) * pq.s
        actual, actualBins = tools.bin_spike_trains(
            {0: [a, b]}, sampling_rate=sampling_rate)
        assert_array_almost_equal(
            expectedBins, actualBins.rescale(expectedBins.units))


if __name__ == '__main__':
    ut.main()
