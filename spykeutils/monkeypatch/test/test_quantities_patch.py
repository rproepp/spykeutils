
try:
    import unittest2 as ut
    assert ut  # Suppress pyflakes warning about redefinition of unused ut
except ImportError:
    import unittest as ut


from itertools import combinations
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


class TestScipyMaximum(ut.TestCase):
    def test_works_with_normal_arrays(self):
        a = sp.array([[4, 2], [2, 4]])
        b = sp.array([[3, 3], [3, 3]])
        expected = sp.array([[4, 3], [3, 4]])
        actual = sp.maximum(a, b)
        assert_array_equal(expected, actual)

    def test_works_with_quantity_arrays(self):
        a = sp.array([[4, 2], [2, 4]]) * pq.s
        b = sp.array([[3000, 3000], [3000, 3000]]) * pq.ms
        expected = sp.array([[4, 3], [3, 4]]) * pq.s
        actual = sp.maximum(a, b)
        assert_array_almost_equal(expected, actual)
        actual = sp.maximum(b, a)
        assert_array_almost_equal(expected, actual)

    def test_works_with_normal_and_quantity_arrays_mixed(self):
        a = sp.array([[4, 2], [2, 4]])
        b = sp.array([[3, 3], [3, 3]]) * pq.dimensionless
        expected = sp.array([[4, 3], [3, 4]]) * pq.dimensionless
        actual = sp.maximum(a, b)
        assert_array_almost_equal(expected, actual)
        actual = sp.maximum(b, a)
        assert_array_almost_equal(expected, actual)

    def test_works_with_(self):
        a = sp.array([[4, 2], [2, 4]]) * pq.dimensionless
        b = sp.array([[3, 3], [3, 3]]) * pq.dimensionless
        expected = sp.array([[4, 3], [3, 4]]) * pq.dimensionless
        actual = sp.maximum(a, b)
        assert_array_almost_equal(expected, actual)
        actual = sp.maximum(b, a)
        assert_array_almost_equal(expected, actual)

    def test_uses_out_param(self):
        a = sp.array([[4, 2], [2, 4]]) * pq.s
        b = sp.array([[3000, 3000], [3000, 3000]]) * pq.ms
        expected = sp.array([[4, 3], [3, 4]]) * pq.s
        out = sp.zeros_like(expected)
        assert out.units == pq.s
        sp.maximum(a, b, out)
        assert_array_almost_equal(expected, out)
        sp.maximum(b, a, out)
        assert_array_almost_equal(expected, out)


class TestScipyMinimum(ut.TestCase):
    def test_works_with_normal_arrays(self):
        a = sp.array([[4, 2], [2, 4]])
        b = sp.array([[3, 3], [3, 3]])
        expected = sp.array([[3, 2], [2, 3]])
        actual = sp.minimum(a, b)
        assert_array_equal(expected, actual)

    def test_works_with_quantity_arrays(self):
        a = sp.array([[4, 2], [2, 4]]) * pq.s
        b = sp.array([[3000, 3000], [3000, 3000]]) * pq.ms
        expected = sp.array([[3, 2], [2, 3]]) * pq.s
        actual = sp.minimum(a, b)
        assert_array_almost_equal(expected, actual)
        actual = sp.minimum(b, a)
        assert_array_almost_equal(expected, actual)

    def test_works_with_normal_and_quantity_arrays_mixed(self):
        a = sp.array([[4, 2], [2, 4]])
        b = sp.array([[3, 3], [3, 3]]) * pq.dimensionless
        expected = sp.array([[3, 2], [2, 3]]) * pq.dimensionless
        actual = sp.minimum(a, b)
        assert_array_almost_equal(expected, actual)
        actual = sp.minimum(b, a)
        assert_array_almost_equal(expected, actual)

    def test_uses_out_param(self):
        a = sp.array([[4, 2], [2, 4]]) * pq.s
        b = sp.array([[3000, 3000], [3000, 3000]]) * pq.ms
        expected = sp.array([[3, 2], [2, 3]]) * pq.s
        out = sp.zeros_like(expected)
        assert out.units == pq.s
        sp.minimum(a, b, out)
        assert_array_almost_equal(expected, out)
        sp.minimum(b, a, out)
        assert_array_almost_equal(expected, out)


class TestScipyMeshgrid(ut.TestCase):
    def test_works_with_normal_arrays(self):
        a = sp.array([1, 2, 3])
        b = sp.array([4, 5])
        expected_a = sp.array([[1, 2, 3], [1, 2, 3]])
        expected_b = sp.array([[4, 4, 4], [5, 5, 5]])
        actual_a, actual_b = sp.meshgrid(a, b)
        assert_array_equal(expected_a, actual_a)
        assert_array_equal(expected_b, actual_b)

    def test_works_with_quantity_arrays(self):
        a = sp.array([1, 2, 3]) * pq.s
        b = sp.array([4, 5]) * pq.m
        expected_a = sp.array([[1, 2, 3], [1, 2, 3]]) * pq.s
        expected_b = sp.array([[4, 4, 4], [5, 5, 5]]) * pq.m
        actual_a, actual_b = sp.meshgrid(a, b)
        assert_array_equal(expected_a, actual_a)
        assert_array_equal(expected_b, actual_b)

    def test_works_with_normal_and_quantity_arrays_mixed(self):
        a = sp.array([1, 2, 3])
        b = sp.array([4, 5]) * pq.m
        expected_a = sp.array([[1, 2, 3], [1, 2, 3]])
        expected_b = sp.array([[4, 4, 4], [5, 5, 5]]) * pq.m
        actual_a, actual_b = sp.meshgrid(a, b)
        assert_array_equal(expected_a, actual_a)
        assert_array_equal(expected_b, actual_b)

        a = sp.array([1, 2, 3]) * pq.s
        b = sp.array([4, 5])
        expected_a = sp.array([[1, 2, 3], [1, 2, 3]]) * pq.s
        expected_b = sp.array([[4, 4, 4], [5, 5, 5]])
        actual_a, actual_b = sp.meshgrid(a, b)
        assert_array_equal(expected_a, actual_a)
        assert_array_equal(expected_b, actual_b)


class TestScipyConcatenate(ut.TestCase):
    def test_works_with_normal_arrays(self):
        a = sp.array([[1]])
        b = sp.array([[2]])
        c = sp.array([[3]])
        axis = 1
        expected = sp.array([[1, 2, 3]])
        actual = sp.concatenate((a, b, c), axis=axis)
        assert_array_equal(expected, actual)

    def test_works_with_quantity_arrays(self):
        a = sp.array([[1]]) * pq.s
        b = sp.array([[2000]]) * pq.ms
        c = sp.array([[3]]) * pq.s
        axis = 1
        expected = sp.array([[1, 2, 3]]) * pq.s
        actual = sp.concatenate((a, b, c), axis=axis)
        assert_array_equal(expected, actual)

    def test_works_with_normal_and_quantity_arrays_mixed(self):
        a = sp.array([[1]])
        b = sp.array([[2]]) * pq.dimensionless
        c = sp.array([[3]])
        axis = 1
        expected = sp.array([[1, 2, 3]])
        actual = sp.concatenate((a, b, c), axis=axis)
        assert_array_equal(expected, actual)

    def test_raises_exception_if_mixing_incompatible_units(self):
        a = sp.array([[1]])
        b = sp.array([[2]]) * pq.dimensionless
        c = sp.array([[3]]) * pq.s
        d = sp.array([[4]]) * pq.m
        for p in combinations((a, c, d), 2):
            with self.assertRaises(Exception):
                sp.concatenate(p)
            with self.assertRaises(Exception):
                sp.concatenate(p[::-1])
        for p in combinations((b, c, d), 2):
            with self.assertRaises(Exception):
                sp.concatenate(p)
            with self.assertRaises(Exception):
                sp.concatenate(p[::-1])

if __name__ == '__main__':
    ut.main()
