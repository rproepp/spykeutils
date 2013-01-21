
try:
    import unittest2 as ut
    assert ut  # Suppress pyflakes warning about redefinition of unused ut
except ImportError:
    import unittest as ut


from itertools import combinations
from numpy.testing import assert_array_equal, assert_array_almost_equal
import scipy as sp
from spykeutils import _scipy_quantities as spq
import quantities as pq


class TestScipyMaximum(ut.TestCase):
    def test_works_with_normal_arrays(self):
        a = sp.array([[4, 2], [2, 4]])
        b = sp.array([[3, 3], [3, 3]])
        expected = sp.array([[4, 3], [3, 4]])
        actual = spq.maximum(a, b)
        assert_array_equal(expected, actual)

    def test_works_with_quantity_arrays(self):
        a = sp.array([[4, 2], [2, 4]]) * pq.s
        b = sp.array([[3000, 3000], [3000, 3000]]) * pq.ms
        expected = sp.array([[4, 3], [3, 4]]) * pq.s
        actual = spq.maximum(a, b)
        assert_array_almost_equal(expected, actual)
        actual = spq.maximum(b, a)
        assert_array_almost_equal(expected, actual)

    def test_works_with_normal_and_quantity_arrays_mixed(self):
        a = sp.array([[4, 2], [2, 4]])
        b = sp.array([[3, 3], [3, 3]]) * pq.dimensionless
        expected = sp.array([[4, 3], [3, 4]]) * pq.dimensionless
        actual = spq.maximum(a, b)
        assert_array_almost_equal(expected, actual)
        actual = spq.maximum(b, a)
        assert_array_almost_equal(expected, actual)

    def test_works_with_(self):
        a = sp.array([[4, 2], [2, 4]]) * pq.dimensionless
        b = sp.array([[3, 3], [3, 3]]) * pq.dimensionless
        expected = sp.array([[4, 3], [3, 4]]) * pq.dimensionless
        actual = spq.maximum(a, b)
        assert_array_almost_equal(expected, actual)
        actual = spq.maximum(b, a)
        assert_array_almost_equal(expected, actual)

    def test_uses_out_param(self):
        a = sp.array([[4, 2], [2, 4]]) * pq.s
        b = sp.array([[3000, 3000], [3000, 3000]]) * pq.ms
        expected = sp.array([[4, 3], [3, 4]]) * pq.s
        out = sp.zeros_like(expected)
        assert out.units == pq.s
        spq.maximum(a, b, out)
        assert_array_almost_equal(expected, out)
        spq.maximum(b, a, out)
        assert_array_almost_equal(expected, out)


class TestScipyMinimum(ut.TestCase):
    def test_works_with_normal_arrays(self):
        a = sp.array([[4, 2], [2, 4]])
        b = sp.array([[3, 3], [3, 3]])
        expected = sp.array([[3, 2], [2, 3]])
        actual = spq.minimum(a, b)
        assert_array_equal(expected, actual)

    def test_works_with_quantity_arrays(self):
        a = sp.array([[4, 2], [2, 4]]) * pq.s
        b = sp.array([[3000, 3000], [3000, 3000]]) * pq.ms
        expected = sp.array([[3, 2], [2, 3]]) * pq.s
        actual = spq.minimum(a, b)
        assert_array_almost_equal(expected, actual)
        actual = spq.minimum(b, a)
        assert_array_almost_equal(expected, actual)

    def test_works_with_normal_and_quantity_arrays_mixed(self):
        a = sp.array([[4, 2], [2, 4]])
        b = sp.array([[3, 3], [3, 3]]) * pq.dimensionless
        expected = sp.array([[3, 2], [2, 3]]) * pq.dimensionless
        actual = spq.minimum(a, b)
        assert_array_almost_equal(expected, actual)
        actual = spq.minimum(b, a)
        assert_array_almost_equal(expected, actual)

    def test_uses_out_param(self):
        a = sp.array([[4, 2], [2, 4]]) * pq.s
        b = sp.array([[3000, 3000], [3000, 3000]]) * pq.ms
        expected = sp.array([[3, 2], [2, 3]]) * pq.s
        out = sp.zeros_like(expected)
        assert out.units == pq.s
        spq.minimum(a, b, out)
        assert_array_almost_equal(expected, out)
        spq.minimum(b, a, out)
        assert_array_almost_equal(expected, out)


class TestScipyMeshgrid(ut.TestCase):
    def test_works_with_normal_arrays(self):
        a = sp.array([1, 2, 3])
        b = sp.array([4, 5])
        expected_a = sp.array([[1, 2, 3], [1, 2, 3]])
        expected_b = sp.array([[4, 4, 4], [5, 5, 5]])
        actual_a, actual_b = spq.meshgrid(a, b)
        assert_array_equal(expected_a, actual_a)
        assert_array_equal(expected_b, actual_b)

    def test_works_with_quantity_arrays(self):
        a = sp.array([1, 2, 3]) * pq.s
        b = sp.array([4, 5]) * pq.m
        expected_a = sp.array([[1, 2, 3], [1, 2, 3]]) * pq.s
        expected_b = sp.array([[4, 4, 4], [5, 5, 5]]) * pq.m
        actual_a, actual_b = spq.meshgrid(a, b)
        assert_array_equal(expected_a, actual_a)
        assert_array_equal(expected_b, actual_b)

    def test_works_with_normal_and_quantity_arrays_mixed(self):
        a = sp.array([1, 2, 3])
        b = sp.array([4, 5]) * pq.m
        expected_a = sp.array([[1, 2, 3], [1, 2, 3]])
        expected_b = sp.array([[4, 4, 4], [5, 5, 5]]) * pq.m
        actual_a, actual_b = spq.meshgrid(a, b)
        assert_array_equal(expected_a, actual_a)
        assert_array_equal(expected_b, actual_b)

        a = sp.array([1, 2, 3]) * pq.s
        b = sp.array([4, 5])
        expected_a = sp.array([[1, 2, 3], [1, 2, 3]]) * pq.s
        expected_b = sp.array([[4, 4, 4], [5, 5, 5]])
        actual_a, actual_b = spq.meshgrid(a, b)
        assert_array_equal(expected_a, actual_a)
        assert_array_equal(expected_b, actual_b)


class TestScipyConcatenate(ut.TestCase):
    def test_works_with_normal_arrays(self):
        a = sp.array([[1]])
        b = sp.array([[2]])
        c = sp.array([[3]])
        axis = 1
        expected = sp.array([[1, 2, 3]])
        actual = spq.concatenate((a, b, c), axis=axis)
        assert_array_equal(expected, actual)

    def test_works_with_quantity_arrays(self):
        a = sp.array([[1]]) * pq.s
        b = sp.array([[2000]]) * pq.ms
        c = sp.array([[3]]) * pq.s
        axis = 1
        expected = sp.array([[1, 2, 3]]) * pq.s
        actual = spq.concatenate((a, b, c), axis=axis)
        assert_array_equal(expected, actual)

    def test_works_with_normal_and_quantity_arrays_mixed(self):
        a = sp.array([[1]])
        b = sp.array([[2]]) * pq.dimensionless
        c = sp.array([[3]])
        axis = 1
        expected = sp.array([[1, 2, 3]])
        actual = spq.concatenate((a, b, c), axis=axis)
        assert_array_equal(expected, actual)

    def test_raises_exception_if_mixing_incompatible_units(self):
        a = sp.array([[1]])
        b = sp.array([[2]]) * pq.dimensionless
        c = sp.array([[3]]) * pq.s
        d = sp.array([[4]]) * pq.m
        for p in combinations((a, c, d), 2):
            with self.assertRaises(Exception):
                spq.concatenate(p)
            with self.assertRaises(Exception):
                spq.concatenate(p[::-1])
        for p in combinations((b, c, d), 2):
            with self.assertRaises(Exception):
                spq.concatenate(p)
            with self.assertRaises(Exception):
                spq.concatenate(p[::-1])


if __name__ == '__main__':
    ut.main()
