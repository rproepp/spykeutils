try:
    import unittest2 as ut
    assert ut # Suppress pyflakes warning about redefinition of unused ut
except ImportError:
    import unittest as ut

import scipy as sp
import quantities as pq
import neo
#from neo.test.tools import assert_arrays_equal
import spykeutils.spike_train_generation as stg


class CommonSpikeTrainGeneratorTests(object):
    """ Provides some common test cases which should work for all spike train
    generation functions.
    """


    def __init__(self, genFunc, defaultRate, lowRate, highRate):
        """
        :param genFunc: The function to call to generate a spike train by the
            test cases.

        Some firing rate values to use in the tests have to be set. They
        will be passed to `genFunc` as first argument.

        :param defaultRate: A default firing rate around 10Hz.
        :param lowRate: A low firing rate around 1Hz.
        :param highRate A high firing rate around 10000Hz.
        """

        self.genFunc = genFunc
        self.defaultRate = defaultRate
        self.lowRate = lowRate
        self.highRate = highRate


    def test_returns_SpikeTrain_containing_spikes(self):
        st = self.genFunc(self.defaultRate, max_spikes=10)
        self.assertIsInstance(st, neo.SpikeTrain)
        self.assertTrue(st.size > 0)


    def test_exception_without_end_condition(self):
        with self.assertRaises(ValueError):
            self.genFunc(self.defaultRate, t_stop=None, max_spikes=None)


    def test_num_spikes_limited_by_max_spike(self):
        max_spikes = 10

        # Use a high rate to provoke more than `max_spike` spikes.
        self.assertEqual(max_spikes,
            self.genFunc(self.highRate, max_spikes=max_spikes).size)

        # Use a long trial length to provoke more than `max_spike` spikes.
        self.assertTrue(max_spikes >=
            self.genFunc(self.lowRate, t_stop=10000 * pq.s,
                max_spikes=max_spikes).size)


    def test_times_limited_by_t_start_and_t_stop(self):
        t_start = 10 * pq.s
        t_stop = 20 * pq.s

        st = self.genFunc(self.defaultRate, t_start=t_start, t_stop=t_stop)
        self.assertTrue(sp.all(t_start < st))
        self.assertTrue(sp.all(st <= t_stop))
        self.assertEqual(t_start, st.t_start)
        self.assertEqual(t_stop, st.t_stop)

        # Use a high value for max_spike to not hit that limit
        st = self.genFunc(self.lowRate, t_start=t_start, t_stop=t_stop,
                max_spikes=10000)
        self.assertTrue(sp.all(t_start < st))
        self.assertTrue(sp.all(st <= t_stop))
        self.assertEqual(t_start, st.t_start)
        self.assertEqual(t_stop, st.t_stop)



class Test_gen_homogeneous_poisson(ut.TestCase, CommonSpikeTrainGeneratorTests):
    def __init__(self, methodName='runTest'):
        ut.TestCase.__init__(self, methodName)
        CommonSpikeTrainGeneratorTests.__init__(self,
                stg.gen_homogeneous_poisson, 10 * pq.Hz, 1 * pq.Hz,
                10000 * pq.Hz)



if __name__ == '__main__':
    ut.main()

