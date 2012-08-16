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
    def setUp(self):
        sp.random.seed(123)
        dimension = 40 # Needs to be divisible by 4 for Spike objects test!
        offset = sp.zeros((dimension,1))
        offset[0] = 4
        self.cluster1 = sp.randn(dimension,10)
        self.cluster2 = sp.randn(dimension,100) + offset
        self.cluster3 = sp.randn(dimension,500) - offset
        self.clusterList1 = [self.cluster1[:,i]
                             for i in xrange(sp.size(self.cluster1,1))]
        self.clusterList2 = [self.cluster2[:,i]
                             for i in xrange(sp.size(self.cluster2,1))]
        self.clusterList3 = [self.cluster3[:,i]
                             for i in xrange(sp.size(self.cluster3,1))]
        self.mean1 = sp.zeros(dimension)
        self.mean2 = offset.flatten()
        self.mean3 = -self.mean2


    def test_refperiod_violations_empty(self):
        num, d = qa.get_refperiod_violations({}, 1*pq.ms)
        self.assertEqual(num, 0)
        self.assertEqual(d, {})


    def test_refperiod_violations(self):
        t1 = sp.array([0, 5, 10, 12, 17, 18])
        t2 = sp.array([20000, 18000, 14000, 9000, 0, 5000])
        t3 = sp.array([1,2,3])
        st1 = neo.SpikeTrain(t1*pq.s,20*pq.s)
        st2 = neo.SpikeTrain(t2*pq.ms,20*pq.s)
        st3 = neo.SpikeTrain(t3*pq.h,3*pq.h)
        r1,r2 = qa.get_refperiod_violations({0: [st1], 1: [st2,st3]},
            3.0*pq.s)

        self.assertEqual(r1, 3,
            'Total number of refractory period violations incorrect')
        assert_arrays_equal(r2[0][0],
            neo.SpikeTrain(sp.array([10,17])*pq.s,20*pq.s))
        assert_arrays_equal(r2[1][0],
            neo.SpikeTrain(sp.array([18])*pq.s,20*pq.s))
        assert_arrays_equal(r2[1][1],
            neo.SpikeTrain(sp.array([])*pq.s,10800*pq.ms))


    def test_refperiod_fp_empty(self):
        self.assertEqual(
            qa.calculate_refperiod_fp({}, 1*pq.ms, {}, 10*pq.ms), {})


    def test_refperiod_fp(self):
        r = qa.calculate_refperiod_fp({1:100, 2:100, 3:100},
            2*pq.ms, {1:19, 2:100, 3:200}, 100*pq.ms)
        self.assertAlmostEqual(r[1], 0.05)
        self.assertAlmostEqual(r[2], 0.5)
        self.assertAlmostEqual(r[3], 1.0)


    def test_overlap_1_cluster(self):
        # One cluster cannot have overlaps...
        total, pair = qa.calculate_overlap_fp_fn(
            {1: self.mean1}, {1: self.clusterList1})
        self.assertEqual(total[1][0], 0.0)
        self.assertEqual(total[1][1], 0.0)
        self.assertEqual(pair, {})


    def test_overlap_2_clusters(self):
        total, pair = qa.calculate_overlap_fp_fn(
                {1: self.mean1, 2: self.mean2},
                {1: self.clusterList1, 2: self.clusterList2})
        self.assertAlmostEqual(total[1][0], 0.122414, 5)
        self.assertAlmostEqual(total[1][1], 0.151077, 5)
        self.assertAlmostEqual(total[2][0], 0.015107, 5)
        self.assertAlmostEqual(total[2][1], 0.012241, 5)
        self.assertAlmostEqual(pair[1][2][0], 0.122414, 5)
        self.assertAlmostEqual(pair[1][2][1], 0.151077, 5)
        self.assertAlmostEqual(pair[2][1][0], 0.015107, 5)
        self.assertAlmostEqual(pair[2][1][1], 0.012241, 5)


    def test_overlap_3_clusters(self):
        total, pair = qa.calculate_overlap_fp_fn(
                {1: self.mean1, 2: self.mean2, 3: self.mean3},
                {1: self.clusterList1, 2: self.clusterList2,
                 3: self.clusterList3})
        self.assertAlmostEqual(total[1][0], 0.424307, 5)
        self.assertAlmostEqual(total[1][1], 0.343706, 5)
        self.assertAlmostEqual(total[2][0], 0.015108, 5)
        self.assertAlmostEqual(total[2][1], 0.012239, 5)
        self.assertAlmostEqual(total[3][0], 0.003852, 5)
        self.assertAlmostEqual(total[3][1], 0.006038, 5)
        self.assertAlmostEqual(pair[1][2][0], 0.122414, 5)
        self.assertAlmostEqual(pair[1][2][1], 0.151077, 5)
        self.assertAlmostEqual(pair[1][3][0], 0.301949, 5)
        self.assertAlmostEqual(pair[1][3][1], 0.192634, 5)
        self.assertAlmostEqual(pair[2][3][0], 1.265653e-06, 11)
        self.assertAlmostEqual(pair[2][3][1], 1.154522e-06, 11)


    def test_overlap_spike_objects(self):
        total, pair = qa.calculate_overlap_fp_fn(
                {1: self.mean1, 2: self.mean2, 3: self.mean3},
                {1: self.clusterList1, 2: self.clusterList2,
                 3: self.clusterList3})

        # Replacing some arrays with Spike objects
        mean2 = neo.Spike(waveform=self.mean2.reshape(-1,4) / 1000.0 * pq.mV)
        mean3 = neo.Spike(waveform=self.mean3.reshape(-1,4) / 1e6 * pq.V)
        clusterList1 = []
        for s in self.clusterList1:
            clusterList1.append(
                neo.Spike(waveform=s.reshape(-1,4) / 1000.0 * pq.mV))

        total_s, pair_s = qa.calculate_overlap_fp_fn(
                {1: self.mean1, 2: mean2, 3: mean3},
                {1: clusterList1, 2: self.clusterList2,
                 3: self.clusterList3})

        # Results should be identical for arrays and Spike objects
        for i in total.keys():
            self.assertAlmostEqual(total[i][0], total_s[i][0])
            self.assertAlmostEqual(total[i][1], total_s[i][1])
            for j in pair[i].keys():
                self.assertAlmostEqual(pair[i][j][0], pair_s[i][j][0])
                self.assertAlmostEqual(pair[i][j][1], pair_s[i][j][1])


if __name__ == '__main__':
    ut.main()