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


if __name__ == '__main__':
    ut.main()
