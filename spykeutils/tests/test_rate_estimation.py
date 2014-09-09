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


class TestScipyMaximum(ut.TestCase):
    def test_collapse_empty_list(self):
        try:
            re.collapsed_spike_trains([])
        except:
            self.fail('Collapsing an empty list of spike trains should'
                      'not raise an exception')

    def test_density_estimation_empty_list(self):
        try:
            re.spike_density_estimation({1: []})
        except:
            self.fail('Density estimation with an empty list of spike'
                      'trains should not raise an exception')

    def test_psth_empty_list(self):
        try:
            re.psth({1: []}, 100 * pq.ms)
        except:
            self.fail('PSTH with an empty list of spikes should not raise'
                      'an exception')


if __name__ == '__main__':
    ut.main()
