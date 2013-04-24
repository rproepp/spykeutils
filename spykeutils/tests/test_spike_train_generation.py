try:
    import unittest2 as ut
    assert ut  # Suppress pyflakes warning about redefinition of unused ut
except ImportError:
    import unittest as ut

import neo
import quantities as pq
import scipy as sp
import spykeutils.spike_train_generation as stg


class CommonSpikeTrainGeneratorTests(object):
    """ Provides some common test cases which should work for all spike train
    generation functions.
    """

    defaultRate = 10 * pq.Hz
    lowRate = 1 * pq.Hz
    highRate = 10000 * pq.Hz

    def invoke_gen_func(self, rate, **kwargs):
        """ This function is called to generate a spike train by the test cases.
        """
        raise NotImplementedError()

    def test_returns_SpikeTrain_containing_spikes(self):
        st = self.invoke_gen_func(self.defaultRate, t_stop=100 * pq.s)
        self.assertIsInstance(st, neo.SpikeTrain)
        self.assertTrue(st.size > 0)

    def test_exception_without_end_condition(self):
        with self.assertRaises(ValueError):
            self.invoke_gen_func(self.defaultRate, t_stop=None)

    def test_times_limited_by_t_start_and_t_stop(self):
        t_start = 10 * pq.s
        t_stop = 20 * pq.s

        st = self.invoke_gen_func(
            self.defaultRate, t_start=t_start, t_stop=t_stop)
        self.assertTrue(sp.all(t_start < st))
        self.assertTrue(sp.all(st <= t_stop))
        self.assertEqual(t_start, st.t_start)
        self.assertEqual(t_stop, st.t_stop)

    def test_num_spikes_limited_by_max_spike(self):
        max_spikes = 10

        # Use a high rate to provoke more than `max_spike` spikes.
        self.assertTrue(
            max_spikes >= self.invoke_gen_func(
                self.highRate, max_spikes=max_spikes).size)

        # Use a long trial length to provoke more than `max_spike` spikes.
        self.assertTrue(
            max_spikes >= self.invoke_gen_func(
                self.lowRate, t_stop=10000 * pq.s, max_spikes=max_spikes).size)

    def test_respects_refractory_period(self):
        refractory = 100 * pq.ms
        st = self.invoke_gen_func(
            self.highRate, max_spikes=1000, refractory=refractory)
        self.assertGreater(
            sp.amax(sp.absolute(sp.diff(st.rescale(pq.s).magnitude))),
            refractory.rescale(pq.s).magnitude)
        st = self.invoke_gen_func(
            self.highRate, t_stop=10 * pq.s, refractory=refractory)
        self.assertGreater(
            sp.amax(sp.absolute(sp.diff(st.rescale(pq.s).magnitude))),
            refractory.rescale(pq.s).magnitude)


class Test_gen_homogeneous_poisson(ut.TestCase, CommonSpikeTrainGeneratorTests):
    def invoke_gen_func(self, rate, **kwargs):
        return stg.gen_homogeneous_poisson(rate, **kwargs)


class Test_gen_inhomogeneous_poisson(
        ut.TestCase, CommonSpikeTrainGeneratorTests):
    def invoke_gen_func(self, max_rate, **kwargs):
        modulation = lambda ts: sp.sin(ts / (5.0 * pq.s) * sp.pi)
        return stg.gen_inhomogeneous_poisson(modulation, max_rate, **kwargs)


if __name__ == '__main__':
    ut.main()
