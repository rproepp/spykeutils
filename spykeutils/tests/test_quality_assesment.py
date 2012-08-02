try:
    import unittest2 as ut
except ImportError:
    import unittest as ut


import scipy as sp
import quantities as pq
import neo
from neo.test.tools import assert_arrays_equal
import spykeutils.sorting_quality_assesment as qa

class TestQualityAssessment(ut.TestCase):
    def test_refperiod_violations(self):
        t1 = sp.array([0, 5, 10, 12, 17, 18])
        t2 = sp.array([20000, 18000, 14000, 9000, 0, 5000])
        t3 = sp.array([1,2,3])
        st1 = neo.SpikeTrain(t1*pq.s,20*pq.s)
        st2 = neo.SpikeTrain(t2*pq.ms,20*pq.s)
        st3 = neo.SpikeTrain(t3*pq.h,3*pq.h)
        r1,r2 = qa.get_refperiod_violations({0: [st1], 1: [st2,st3]}, 3.0*pq.s)

        self.assertEqual(r1, 3, 'Total number of refractory period violations incorrect')
        assert_arrays_equal(r2[0][0], neo.SpikeTrain(sp.array([10,17])*pq.s,20*pq.s))
        assert_arrays_equal(r2[1][0], neo.SpikeTrain(sp.array([18])*pq.s,20*pq.s))
        assert_arrays_equal(r2[1][1], neo.SpikeTrain(sp.array([])*pq.s,10800*pq.ms))

    def test_refperiod_fp(self):
        r = qa.calculate_refperiod_fp({1:100, 2:100, 3:100}, 2*pq.ms, {1:19, 2:100, 3:200}, 100*pq.ms)
        self.assertAlmostEqual(r[1], 0.05)
        self.assertAlmostEqual(r[2], 0.5)
        self.assertAlmostEqual(r[3], 1.0)

if __name__ == '__main__':
    ut.main()