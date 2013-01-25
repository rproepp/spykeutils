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
        cluster1 = sp.randn(8,100)
        clusterList1 = [cluster1[:,i]
                        for i in xrange(sp.size(cluster1,1))]
        total, pair = qa.overlap_fp_fn({1: clusterList1})
        self.assertEqual(total[1][0], 0.0)
        self.assertEqual(total[1][1], 0.0)
        self.assertEqual(pair, {})

    def test_overlap_equal_clusters_white(self):
        cluster1 = sp.randn(40,1000)
        cluster2 = sp.randn(40,1000)
        clusterList1 = [cluster1[:,i]
                             for i in xrange(sp.size(cluster1,1))]
        clusterList2 = [cluster2[:,i]
                             for i in xrange(sp.size(cluster2,1))]
        total, pair = qa.overlap_fp_fn(
            {1: clusterList1, 2: clusterList2},
            means={1: sp.zeros(40), 2: sp.zeros(40)},
            covariances='white')
        self.assertAlmostEqual(total[1][0], 0.5)
        self.assertAlmostEqual(total[1][1], 0.5)
        self.assertAlmostEqual(total[2][0], 0.5)
        self.assertAlmostEqual(total[2][1], 0.5)
        self.assertAlmostEqual(pair[1][2][0], 0.5)
        self.assertAlmostEqual(pair[1][2][1], 0.5)
        self.assertAlmostEqual(pair[2][1][0], 0.5)
        self.assertAlmostEqual(pair[2][1][1], 0.5)

    def test_overlap_equal_clusters_estimate_mean(self):
        # Smaller dimensionality and more data for reliable estimates
        cluster1 = sp.randn(8,100000)
        cluster2 = sp.randn(8,100000)
        clusterList1 = [cluster1[:,i]
                        for i in xrange(sp.size(cluster1,1))]
        clusterList2 = [cluster2[:,i]
                        for i in xrange(sp.size(cluster2,1))]
        total, pair = qa.overlap_fp_fn(
            {1: clusterList1, 2: clusterList2},
            means={1: sp.zeros(8), 2: sp.zeros(8)})
        self.assertAlmostEqual(total[1][0], 0.5, 3)
        self.assertAlmostEqual(total[1][1], 0.5, 3)
        self.assertAlmostEqual(total[2][0], 0.5, 3)
        self.assertAlmostEqual(total[2][1], 0.5, 3)
        self.assertAlmostEqual(pair[1][2][0], 0.5, 3)
        self.assertAlmostEqual(pair[1][2][1], 0.5, 3)
        self.assertAlmostEqual(pair[2][1][0], 0.5, 3)
        self.assertAlmostEqual(pair[2][1][1], 0.5, 3)

    def test_overlap_equal_clusters_estimate_all(self):
        # Smaller dimensionality and more data for reliable estimates
        cluster1 = sp.randn(8,100000)
        cluster2 = sp.randn(8,100000)
        clusterList1 = [cluster1[:,i]
                        for i in xrange(sp.size(cluster1,1))]
        clusterList2 = [cluster2[:,i]
                        for i in xrange(sp.size(cluster2,1))]
        total, pair = qa.overlap_fp_fn(
            {1: clusterList1, 2: clusterList2})
        self.assertAlmostEqual(total[1][0], 0.5, 2)
        self.assertAlmostEqual(total[1][1], 0.5, 2)
        self.assertAlmostEqual(total[2][0], 0.5, 2)
        self.assertAlmostEqual(total[2][1], 0.5, 2)
        self.assertAlmostEqual(pair[1][2][0], 0.5, 2)
        self.assertAlmostEqual(pair[1][2][1], 0.5, 2)
        self.assertAlmostEqual(pair[2][1][0], 0.5, 2)
        self.assertAlmostEqual(pair[2][1][1], 0.5, 2)

    def test_overlap_unequal_clusters(self):
        cluster1 = sp.randn(40,1000)
        cluster2 = sp.randn(40,2000)
        clusterList1 = [cluster1[:,i]
                        for i in xrange(sp.size(cluster1,1))]
        clusterList2 = [cluster2[:,i]
                        for i in xrange(sp.size(cluster2,1))]
        total, pair = qa.overlap_fp_fn(
            {1: clusterList1, 2: clusterList2},
            {1: sp.zeros(40), 2: sp.zeros(40)},
            {1: sp.eye(40), 2: sp.eye(40)})
        self.assertAlmostEqual(total[1][0], 2.0/3.0)
        self.assertAlmostEqual(total[1][1], 2.0/3.0)
        self.assertAlmostEqual(total[2][0], 1.0/3.0)
        self.assertAlmostEqual(total[2][1], 1.0/3.0)
        self.assertAlmostEqual(pair[1][2][0], 2.0/3.0)
        self.assertAlmostEqual(pair[1][2][1], 2.0/3.0)
        self.assertAlmostEqual(pair[2][1][0], 1.0/3.0)
        self.assertAlmostEqual(pair[2][1][1], 1.0/3.0)

    def test_far_apart_clusters_estimate_all(self):
        cluster1 = sp.randn(40,1000)
        cluster2 = sp.randn(40,1000) * 2
        cluster2[0,:] += 10
        clusterList1 = [cluster1[:,i]
                        for i in xrange(sp.size(cluster1,1))]
        clusterList2 = [cluster2[:,i]
                        for i in xrange(sp.size(cluster2,1))]
        total, pair = qa.overlap_fp_fn(
            {1: clusterList1, 2: clusterList2})
        self.assertLess(total[1][0], 1e-4)
        self.assertLess(total[1][1], 1e-4)
        self.assertLess(total[2][0], 1e-4)
        self.assertLess(total[2][1], 1e-4)
        self.assertLess(pair[1][2][0], 1e-4)
        self.assertLess(pair[1][2][1], 1e-4)
        self.assertLess(pair[2][1][0], 1e-4)
        self.assertLess(pair[2][1][1], 1e-4)
        self.assertGreater(total[1][0], 0.0)
        self.assertGreater(total[1][1], 0.0)
        self.assertGreater(total[2][0], 0.0)
        self.assertGreater(total[2][1], 0.0)
        self.assertGreater(pair[1][2][0], 0.0)
        self.assertGreater(pair[1][2][1], 0.0)
        self.assertGreater(pair[2][1][0], 0.0)
        self.assertGreater(pair[2][1][1], 0.0)

    def test_overlap_3_clusters_estimate_means(self):
        cluster1 = sp.randn(20,10000)
        cluster2 = sp.randn(20,20000)
        cluster3 = sp.randn(20,10000)
        cluster3[5,:] += 11
        clusterList1 = [cluster1[:,i]
                        for i in xrange(sp.size(cluster1,1))]
        clusterList2 = [cluster2[:,i]
                        for i in xrange(sp.size(cluster2,1))]
        clusterList3 = [cluster3[:,i]
                        for i in xrange(sp.size(cluster3,1))]
        total, pair = qa.overlap_fp_fn(
            {1: clusterList1, 2: clusterList2, 3: clusterList3},
            covariances={1:sp.eye(20), 2:sp.eye(20), 3:sp.eye(20)*1.5})

        self.assertAlmostEqual(total[1][0], 2.0/3.0, 2)
        self.assertAlmostEqual(total[1][1], 2.0/3.0, 2)
        self.assertAlmostEqual(total[2][0], 1.0/3.0, 2)
        self.assertAlmostEqual(total[2][1], 1.0/3.0, 2)
        self.assertLess(total[3][0], 1e-4)
        self.assertLess(total[3][1], 1e-4)
        self.assertGreater(total[3][0], 0.0)
        self.assertGreater(total[3][1], 0.0)
        self.assertAlmostEqual(pair[1][2][0], 2.0/3.0, 2)
        self.assertAlmostEqual(pair[1][2][1], 2.0/3.0, 2)
        self.assertLess(pair[1][3][0], 1e-4)
        self.assertLess(pair[1][3][1], 1e-4)
        self.assertGreater(pair[1][3][0], 0.0)
        self.assertGreater(pair[1][3][1], 0.0)
        self.assertAlmostEqual(pair[2][1][0], 1.0/3.0, 2)
        self.assertAlmostEqual(pair[2][1][1], 1.0/3.0, 2)
        self.assertLess(pair[2][3][0], 1e-4)
        self.assertLess(pair[2][3][1], 1e-4)
        self.assertGreater(pair[2][3][0], 0.0)
        self.assertGreater(pair[2][3][1], 0.0)
        self.assertLess(pair[3][1][0], 1e-4)
        self.assertLess(pair[3][1][1], 1e-4)
        self.assertGreater(pair[3][1][0], 0.0)
        self.assertGreater(pair[3][1][1], 0.0)
        self.assertLess(pair[3][2][0], 1e-4)
        self.assertLess(pair[3][2][1], 1e-4)
        self.assertGreater(pair[3][2][0], 0.0)
        self.assertGreater(pair[3][2][1], 0.0)

    def test_overlap_spike_objects(self):
        dimension = 40
        offset = sp.zeros((dimension,1))
        offset[0] = 4
        cluster1 = sp.randn(dimension,10)
        cluster2 = sp.randn(dimension,100) + offset
        cluster3 = sp.randn(dimension,500) - offset
        clusterList1 = [cluster1[:,i]
                             for i in xrange(sp.size(cluster1,1))]
        clusterList2 = [cluster2[:,i]
                             for i in xrange(sp.size(cluster2,1))]
        clusterList3 = [cluster3[:,i]
                             for i in xrange(sp.size(cluster3,1))]
        mean1 = sp.zeros(dimension)
        mean2 = offset.flatten()
        mean3 = -mean2
        
        total, pair = qa.overlap_fp_fn(
            {1: clusterList1, 2: clusterList2, 3: clusterList3},
            means={1: mean1, 2: mean2, 3: mean3},
            covariances={1: sp.eye(dimension), 2: sp.eye(dimension),
                         3: sp.eye(dimension)})

        # Replacing some arrays with Spike objects
        mean2 = neo.Spike(waveform=mean2.reshape(-1,4) / 1000.0 * pq.mV)
        mean3 = neo.Spike(waveform=mean3.reshape(-1,4) / 1e6 * pq.V)
        newClusterList = []
        for s in clusterList1:
            newClusterList.append(
                neo.Spike(waveform=s.reshape(-1,4) / 1000.0 * pq.mV))

        total_s, pair_s = qa.overlap_fp_fn(
                {1: newClusterList, 2: clusterList2, 3: clusterList3},
                means={1: mean1, 2: mean2, 3: mean3},
                covariances='white')

        # Results should be identical for arrays and Spike objects
        for i in total.keys():
            self.assertAlmostEqual(total[i][0], total_s[i][0])
            self.assertAlmostEqual(total[i][1], total_s[i][1])
            for j in pair[i].keys():
                self.assertAlmostEqual(pair[i][j][0], pair_s[i][j][0])
                self.assertAlmostEqual(pair[i][j][1], pair_s[i][j][1])


if __name__ == '__main__':
    ut.main()