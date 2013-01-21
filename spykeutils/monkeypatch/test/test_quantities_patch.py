
try:
    import unittest2 as ut
    assert ut  # Suppress pyflakes warning about redefinition of unused ut
except ImportError:
    import unittest as ut


from numpy.testing import assert_array_equal, assert_array_almost_equal
import scipy as sp
import quantities as pq
from spykeutils.monkeypatch import quantities_patch

assert quantities_patch  # Suppress pyflakes warning, patch applied by loading


class TestQuantityMax(ut.TestCase):
    def test_returns_global_max(self):
        a = sp.array([[1, 2], [3, 4]]) * pq.s
        self.assertEqual(4, a.max())

    def test_returns_axiswise_max(self):
        a = sp.array([[1, 2], [3, 4]]) * pq.s
        axis = 1
        expected = sp.array([2, 4]) * pq.s
        assert_array_equal(expected, a.max(axis=axis))

    def test_returns_result_in_out_array(self):
        a = sp.array([[1, 2], [3, 4]]) * pq.s
        out = 0 * pq.s
        expected = sp.array([4])
        a.max(out=out)
        self.assertEqual(expected, out)


class TestQuantityMin(ut.TestCase):
    def test_returns_global_min(self):
        a = sp.array([[1, 2], [3, 4]]) * pq.s
        self.assertEqual(1, a.min())

    def test_returns_axiswise_min(self):
        a = sp.array([[1, 2], [3, 4]]) * pq.s
        axis = 1
        expected = sp.array([1, 3]) * pq.s
        assert_array_equal(expected, a.min(axis=axis))

    def test_returns_result_in_out_array(self):
        a = sp.array([[1, 2], [3, 4]]) * pq.s
        out = 0 * pq.s
        expected = 1 * pq.s
        a.min(out=out)
        self.assertEqual(expected, out)


if __name__ == '__main__':
    ut.main()
