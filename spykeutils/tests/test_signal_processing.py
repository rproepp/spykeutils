
try:
    import unittest2 as ut
    assert ut  # Suppress pyflakes warning about redefinition of unused ut
except ImportError:
    import unittest as ut

from builders import create_empty_spike_train
import neo
import quantities as pq
import scipy as sp
import spykeutils.signal_processing as sigproc


class Test_st_convolve(ut.TestCase):
    def test_convolution_with_empty_spike_train_returns_array_of_zeros(self):
        st = create_empty_spike_train()
        result = sigproc.st_convolve(st, sigproc.gauss_kernel)
        self.assertTrue(sp.all(result == 0.0))

    def test_length_of_returned_array_equals_sampling_rate_times_duration(self):
        start = 2.0 * pq.s
        stop = 5.0 * pq.s
        duration = stop - start
        sampling_rate = 12 * pq.Hz
        expected_length = (sampling_rate * duration).simplified

        st = create_empty_spike_train(start, stop)
        result = sigproc.st_convolve(
            st, sigproc.gauss_kernel, sampling_rate=sampling_rate)
        self.assertEqual(expected_length, result.size)

    def test_returns_convolved_spike_train(self):
        st = neo.SpikeTrain(sp.array([1.0, 2.0]) * pq.s, t_stop=3.0 * pq.s)
        expected = sp.array(
            [0.0, 0.0, 0.0, 0.0, 1.4444444, 1.4444444, 1.4444444, 0.0,
             1.4444444, 1.4444444, 1.4444444, 0.0])
        actual = sigproc.st_convolve(
            st, sigproc.rectangular_kernel, sampling_rate=4 * pq.Hz,
            half_width=0.3 * pq.s)
        self.assertTrue(sp.all(sp.absolute(expected - actual) < 1e7))

    def test_uses_sampling_rate_of_spike_train_if_none_is_passed(self):
        start = 2.0 * pq.s
        stop = 5.0 * pq.s
        duration = stop - start
        sampling_rate = 12 * pq.Hz
        expected_length = (sampling_rate * duration).simplified

        st = create_empty_spike_train(start, stop)
        st.sampling_rate = sampling_rate
        result = sigproc.st_convolve(st, sigproc.gauss_kernel)
        self.assertEqual(expected_length, result.size)


if __name__ == '__main__':
    ut.main()
