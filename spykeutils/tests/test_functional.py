
try:
    import unittest2 as ut
    assert ut  # Suppress pyflakes warning about redefinition of unused ut
except ImportError:
    import unittest as ut

from builders import arange_spikes
from spykeutils import functional
import quantities as pq


class Test_apply_to_dict(ut.TestCase):
    @staticmethod
    def fn(train, multiplier=1):
        return multiplier * train.size

    def test_maps_function_to_each_spike_train(self):
        st_dict = {'a': [arange_spikes(5 * pq.s), arange_spikes(4 * pq.s)],
                   'b': [arange_spikes(7 * pq.s)]}
        expected = {'a': [4, 3], 'b': [6]}
        actual = functional.apply_to_dict(self.fn, st_dict)
        self.assertEqual(expected, actual)

    def test_works_on_empty_lists(self):
        st_dict = {'a': [], 'b': []}
        expected = {'a': [], 'b': []}
        actual = functional.apply_to_dict(self.fn, st_dict)
        self.assertEqual(expected, actual)

    def test_works_on_empty_dict(self):
        st_dict = {}
        expected = {}
        actual = functional.apply_to_dict(self.fn, st_dict)
        self.assertEqual(expected, actual)

    def test_allows_to_pass_additional_args(self):
        st_dict = {'a': [arange_spikes(5 * pq.s), arange_spikes(4 * pq.s)],
                   'b': [arange_spikes(7 * pq.s)]}
        expected = {'a': [8, 6], 'b': [12]}
        actual = functional.apply_to_dict(self.fn, st_dict, 2)
        self.assertEqual(expected, actual)


if __name__ == '__main__':
    ut.main()
