try:
    import unittest2 as ut
    assert ut  # Suppress pyflakes warning about redefinition of unused ut
except ImportError:
    import unittest as ut

from builders import arange_spikes
from numpy.testing import assert_array_equal, assert_array_almost_equal
from spykeutils import tools
import neo
import neo.io.tools
import neo.test.tools
import quantities as pq
import scipy as sp


class TestApplyToDict(ut.TestCase):
    @staticmethod
    def fn(train, multiplier=1):
        return multiplier * train.size

    def test_maps_function_to_each_spike_train(self):
        st_dict = {'a': [arange_spikes(5 * pq.s), arange_spikes(4 * pq.s)],
                   'b': [arange_spikes(7 * pq.s)]}
        expected = {'a': [4, 3], 'b': [6]}
        actual = tools.apply_to_dict(self.fn, st_dict)
        self.assertEqual(expected, actual)

    def test_works_on_empty_lists(self):
        st_dict = {'a': [], 'b': []}
        expected = {'a': [], 'b': []}
        actual = tools.apply_to_dict(self.fn, st_dict)
        self.assertEqual(expected, actual)

    def test_works_on_empty_dict(self):
        st_dict = {}
        expected = {}
        actual = tools.apply_to_dict(self.fn, st_dict)
        self.assertEqual(expected, actual)

    def test_allows_to_pass_additional_args(self):
        st_dict = {'a': [arange_spikes(5 * pq.s), arange_spikes(4 * pq.s)],
                   'b': [arange_spikes(7 * pq.s)]}
        expected = {'a': [8, 6], 'b': [12]}
        actual = tools.apply_to_dict(self.fn, st_dict, 2)
        self.assertEqual(expected, actual)


class TestBinSpikeTrains(ut.TestCase):
    def test_bins_spike_train_using_its_properties(self):
        a = neo.SpikeTrain(
            sp.array([1000.0]) * pq.ms, t_start=500.0 * pq.ms,
            t_stop=1500.0 * pq.ms)
        sampling_rate = 4.0 * pq.Hz
        expected = {0: [sp.array([0, 0, 1, 0])]}
        expectedBins = sp.array([0.5, 0.75, 1.0, 1.25, 1.5]) * pq.s
        actual, actualBins = tools.bin_spike_trains({0: [a]}, sampling_rate)
        self.assertEqual(len(expected), len(actual))
        self.assertEqual(len(expected[0]), len(actual[0]))
        assert_array_equal(expected[0][0], actual[0][0])
        assert_array_almost_equal(
            expectedBins, actualBins.rescale(expectedBins.units))

    def test_bins_spike_train_using_passed_properties(self):
        a = neo.SpikeTrain(
            sp.array([1.0]) * pq.s, t_start=0.0 * pq.s, t_stop=5.0 * pq.s)
        sampling_rate = 4.0 * pq.Hz
        t_start = 0.5 * pq.s
        t_stop = 1.5 * pq.s
        expected = {0: [sp.array([0, 0, 1, 0])]}
        expectedBins = sp.array([0.5, 0.75, 1.0, 1.25, 1.5]) * pq.s
        actual, actualBins = tools.bin_spike_trains(
            {0: [a]}, sampling_rate=sampling_rate, t_start=t_start,
            t_stop=t_stop)
        self.assertEqual(len(expected), len(actual))
        self.assertEqual(len(expected[0]), len(actual[0]))
        assert_array_equal(expected[0][0], actual[0][0])
        assert_array_almost_equal(
            expectedBins, actualBins.rescale(expectedBins.units))

    def test_uses_max_spike_train_interval(self):
        a = arange_spikes(5 * pq.s)
        b = arange_spikes(7 * pq.s, 15 * pq.s)
        sampling_rate = 4.0 * pq.Hz
        expectedBins = sp.arange(0.0, 15.1, 0.25) * pq.s
        actual, actualBins = tools.bin_spike_trains(
            {0: [a, b]}, sampling_rate=sampling_rate)
        assert_array_almost_equal(
            expectedBins, actualBins.rescale(expectedBins.units))

    def test_handles_bin_size_which_is_not_divisor_of_duration(self):
        a = arange_spikes(5 * pq.s)
        sampling_rate = 1.0 / 1.3 * pq.Hz
        expected = {0: [sp.array([1, 1, 1, 1])]}
        expectedBins = sp.array([0.0, 1.3, 2.6, 3.9, 5.2]) * pq.s
        actual, actualBins = tools.bin_spike_trains({0: [a]}, sampling_rate)
        self.assertEqual(len(expected), len(actual))
        self.assertEqual(len(expected[0]), len(actual[0]))
        assert_array_equal(expected[0][0], actual[0][0])
        assert_array_almost_equal(
            expectedBins, actualBins.rescale(expectedBins.units))


class TestConcatenateSpikeTrains(ut.TestCase):
    def test_concatenates_spike_trains(self):
        a = arange_spikes(3.0 * pq.s)
        b = arange_spikes(2.0 * pq.s, 5.0 * pq.s)
        expected = arange_spikes(5.0 * pq.s)
        actual = tools.concatenate_spike_trains((a, b))
        assert_array_almost_equal(expected, actual)

    def test_t_start_is_min_of_all_trains(self):
        a = arange_spikes(3.0 * pq.s, 5.0 * pq.s)
        b = arange_spikes(1.0 * pq.s, 6.0 * pq.s)
        expected = 1.0 * pq.s
        actual = tools.concatenate_spike_trains((a, b)).t_start
        self.assertAlmostEqual(expected, actual)

    def test_t_stop_is_max_of_all_trains(self):
        a = arange_spikes(3.0 * pq.s, 5.0 * pq.s)
        b = arange_spikes(1.0 * pq.s, 6.0 * pq.s)
        expected = 6.0 * pq.s
        actual = tools.concatenate_spike_trains((a, b)).t_stop
        self.assertAlmostEqual(expected, actual)


class TestRemoveFromHierarchy(ut.TestCase):
    SEGMENTS = 5
    CHANNEL_GROUPS = 4
    UNITS = 3
    CHANNELS = 4

    @classmethod
    def create_hierarchy(cls, many_to_many):
        b = neo.Block()

        for ns in range(cls.SEGMENTS):
            b.segments.append(neo.Segment())

        channels = []
        if many_to_many:
            channels = [neo.RecordingChannel(name='Shared %d' % i,
                                             index=i + cls.CHANNELS)
                        for i in range(cls.CHANNELS / 2)]

        for ng in range(cls.CHANNEL_GROUPS):
            rcg = neo.RecordingChannelGroup()
            for nu in range(cls.UNITS):
                unit = neo.Unit()
                for ns in range(cls.SEGMENTS):
                    spike = neo.Spike(0 * pq.s)
                    unit.spikes.append(spike)
                    b.segments[ns].spikes.append(spike)

                    st = neo.SpikeTrain([] * pq.s, 0 * pq.s)
                    unit.spiketrains.append(st)
                    b.segments[ns].spiketrains.append(st)

                rcg.units.append(unit)

            if not many_to_many:
                for nc in range(cls.CHANNELS):
                    rc = neo.RecordingChannel(
                        name='Single %d' % nc, index=nc)
                    rc.recordingchannelgroups.append(rcg)
                    rcg.recordingchannels.append(rc)
            else:
                for nc in range(cls.CHANNELS):
                    if nc % 2 == 0:
                        rc = neo.RecordingChannel(
                            name='Single %d' % (nc / 2), index=nc / 2)
                    else:
                        rc = channels[nc / 2]
                    rc.recordingchannelgroups.append(rcg)
                    rcg.recordingchannels.append(rc)
            rcg.channel_indexes = sp.array(
                [c.index for c in rcg.recordingchannels])
            rcg.channel_names = sp.array(
                [c.name for c in rcg.recordingchannels])

            b.recordingchannelgroups.append(rcg)

        try:
            neo.io.tools.create_many_to_one_relationship(b)
        except AttributeError:
            b.create_many_to_one_relationship()
        return b

    def test_remove_block(self):
        block = self.create_hierarchy(False)
        comp = self.create_hierarchy(False)
        tools.remove_from_hierarchy(block)

        neo.test.tools.assert_same_sub_schema(block, comp)

    def test_remove_segment_no_orphans(self):
        block = self.create_hierarchy(False)
        comp = self.create_hierarchy(False)

        seg = block.segments[1]
        tools.remove_from_hierarchy(seg)

        self.assertFalse(seg in block.segments)
        self.assertEqual(len(block.list_units),
                         self.UNITS * self.CHANNEL_GROUPS)
        for u in block.list_units:
            self.assertEqual(len(u.spikes), self.SEGMENTS - 1)
            self.assertEqual(len(u.spiketrains), self.SEGMENTS - 1)
        neo.test.tools.assert_same_sub_schema(seg, comp.segments[1])

    def test_remove_segment_keep_orphans(self):
        block = self.create_hierarchy(False)
        comp = self.create_hierarchy(False)

        seg = block.segments[1]
        tools.remove_from_hierarchy(seg, False)

        self.assertFalse(seg in block.segments)
        self.assertEqual(len(block.list_units),
                         self.UNITS * self.CHANNEL_GROUPS)
        for u in block.list_units:
            self.assertEqual(len(u.spikes), self.SEGMENTS)
            self.assertEqual(len(u.spiketrains), self.SEGMENTS)
        neo.test.tools.assert_same_sub_schema(seg, comp.segments[1])

    def test_remove_channel_group_no_orphans(self):
        block = self.create_hierarchy(False)
        comp = self.create_hierarchy(False)

        rcg = block.recordingchannelgroups[1]
        tools.remove_from_hierarchy(rcg)

        self.assertFalse(rcg in block.recordingchannelgroups)
        self.assertEqual(len(block.segments), self.SEGMENTS)
        for s in block.segments:
            self.assertEqual(len(s.spikes),
                             self.UNITS * (self.CHANNEL_GROUPS - 1))
            self.assertEqual(len(s.spiketrains),
                             self.UNITS * (self.CHANNEL_GROUPS - 1))
        neo.test.tools.assert_same_sub_schema(rcg,
                                              comp.recordingchannelgroups[1])

    def test_remove_channel_group_keep_orphans(self):
        block = self.create_hierarchy(False)
        comp = self.create_hierarchy(False)

        rcg = block.recordingchannelgroups[1]
        tools.remove_from_hierarchy(rcg, False)

        self.assertFalse(rcg in block.recordingchannelgroups)
        self.assertEqual(len(block.segments), self.SEGMENTS)
        for s in block.segments:
            self.assertEqual(len(s.spikes),
                             self.UNITS * self.CHANNEL_GROUPS)
            self.assertEqual(len(s.spiketrains),
                             self.UNITS * self.CHANNEL_GROUPS)
        neo.test.tools.assert_same_sub_schema(rcg,
                                              comp.recordingchannelgroups[1])

    def test_remove_channel(self):
        block = self.create_hierarchy(False)
        comp = self.create_hierarchy(False)

        rc = block.list_recordingchannels[5]
        tools.remove_from_hierarchy(rc)
        self.assertFalse(rc in block.list_recordingchannels)
        neo.test.tools.assert_same_sub_schema(rc,
                                              comp.list_recordingchannels[5])

        self.assertEqual(len(block.segments), self.SEGMENTS)
        self.assertEqual(len(block.recordingchannelgroups),
                         self.CHANNEL_GROUPS)
        self.assertEqual(len(block.list_recordingchannels),
                         self.CHANNEL_GROUPS * self.CHANNELS - 1)

        # Should be removed from its own channel group
        rcg = rc.recordingchannelgroups[0]
        self.assertEqual(len(rcg.recordingchannels), self.CHANNELS - 1)
        self.assertEqual(rcg.channel_indexes.shape[0], self.CHANNELS - 1)
        self.assertEqual(rcg.channel_names.shape[0], self.CHANNELS - 1)
        self.assertFalse(rc.index in rcg.channel_indexes)
        self.assertFalse(rc.name in rcg.channel_names)

    def test_remove_unique_channel_many_to_many(self):
        block = self.create_hierarchy(True)
        comp = self.create_hierarchy(True)
        self.assertEqual(
            len(block.list_recordingchannels),
            self.CHANNEL_GROUPS * (self.CHANNELS / 2) + (self.CHANNELS / 2))

        rc = block.list_recordingchannels[0]  # Unique channel
        tools.remove_from_hierarchy(rc)

        neo.test.tools.assert_same_sub_schema(rc,
                                              comp.list_recordingchannels[0])
        self.assertFalse(rc in block.list_recordingchannels)

        self.assertEqual(len(block.segments), self.SEGMENTS)
        self.assertEqual(len(block.recordingchannelgroups),
                         self.CHANNEL_GROUPS)
        self.assertEqual(
            len(block.list_recordingchannels),
            self.CHANNEL_GROUPS * (self.CHANNELS / 2) + (self.CHANNELS / 2) - 1)

        # Should be removed from its own channel group
        rcg = rc.recordingchannelgroups[0]
        self.assertEqual(len(rcg.recordingchannels), self.CHANNELS - 1)
        self.assertEqual(rcg.channel_indexes.shape[0], self.CHANNELS - 1)
        self.assertEqual(rcg.channel_names.shape[0], self.CHANNELS - 1)
        self.assertFalse(rc.index in rcg.channel_indexes)
        self.assertFalse(rc.name in rcg.channel_names)

    def test_remove_shared_channel_many_to_many(self):
        block = self.create_hierarchy(True)
        comp = self.create_hierarchy(True)
        self.assertEqual(
            len(block.list_recordingchannels),
            self.CHANNEL_GROUPS * (self.CHANNELS / 2) + (self.CHANNELS / 2))

        rc = block.list_recordingchannels[1]  # Shared channel
        tools.remove_from_hierarchy(rc)

        neo.test.tools.assert_same_sub_schema(rc,
                                              comp.list_recordingchannels[1])
        self.assertFalse(rc in block.list_recordingchannels)

        self.assertEqual(len(block.segments), self.SEGMENTS)
        self.assertEqual(len(block.recordingchannelgroups),
                         self.CHANNEL_GROUPS)
        self.assertEqual(
            len(block.list_recordingchannels),
            self.CHANNEL_GROUPS * (self.CHANNELS / 2) + (self.CHANNELS / 2) - 1)

        # Should be removed from all channel groups
        for rcg in block.recordingchannelgroups:
            self.assertEqual(len(rcg.recordingchannels), self.CHANNELS - 1)
            self.assertEqual(rcg.channel_indexes.shape[0], self.CHANNELS - 1)
            self.assertEqual(rcg.channel_names.shape[0], self.CHANNELS - 1)
            self.assertFalse(rc.index in rcg.channel_indexes)
            self.assertFalse(rc.name in rcg.channel_names)

    def test_remove_unit_no_orphans(self):
        block = self.create_hierarchy(False)
        comp = self.create_hierarchy(False)

        unit = block.list_units[5]
        tools.remove_from_hierarchy(unit)

        self.assertFalse(unit in block.list_units)
        self.assertEqual(len(block.list_units),
                         self.UNITS * self.CHANNEL_GROUPS - 1)
        self.assertEqual(len(block.segments), self.SEGMENTS)
        self.assertEqual(len(block.recordingchannelgroups),
                         self.CHANNEL_GROUPS)
        for seg in block.segments:
            self.assertEqual(len(seg.spikes),
                             self.UNITS * self.CHANNEL_GROUPS - 1)
            self.assertEqual(len(seg.spiketrains),
                             self.UNITS * self.CHANNEL_GROUPS - 1)
            self.assertFalse(unit in [s.unit for s in seg.spikes])
            self.assertFalse(unit in [st.unit for st in seg.spiketrains])
        neo.test.tools.assert_same_sub_schema(unit, comp.list_units[5])

    def test_remove_unit_keep_orphans(self):
        block = self.create_hierarchy(False)
        comp = self.create_hierarchy(False)

        unit = block.list_units[5]
        tools.remove_from_hierarchy(unit, False)

        self.assertFalse(unit in block.list_units)
        self.assertEqual(len(block.list_units),
                         self.UNITS * self.CHANNEL_GROUPS - 1)
        self.assertEqual(len(block.segments), self.SEGMENTS)
        self.assertEqual(len(block.recordingchannelgroups),
                         self.CHANNEL_GROUPS)
        for seg in block.segments:
            self.assertEqual(len(seg.spikes),
                             self.UNITS * self.CHANNEL_GROUPS)
            self.assertEqual(len(seg.spiketrains),
                             self.UNITS * self.CHANNEL_GROUPS)
            self.assertFalse(unit in [s.unit for s in seg.spikes])
            self.assertFalse(unit in [st.unit for st in seg.spiketrains])
        neo.test.tools.assert_same_sub_schema(unit, comp.list_units[5])

    def test_remove_spike(self):
        unit = neo.Unit()
        segment = neo.Segment()

        s = neo.Spike(0 * pq.s)
        unit.spikes.append(s)
        segment.spikes.append(s)
        s.unit = unit
        s.segment = segment

        st = neo.SpikeTrain([] * pq.s, 0 * pq.s)
        unit.spiketrains.append(st)
        segment.spiketrains.append(st)
        st.unit = unit
        st.segment = segment

        tools.remove_from_hierarchy(s)
        self.assertTrue(st in unit.spiketrains)
        self.assertTrue(st in segment.spiketrains)
        self.assertFalse(s in unit.spikes)
        self.assertFalse(s in segment.spikes)

    def test_remove_spiketrain(self):
        unit = neo.Unit()
        segment = neo.Segment()

        s = neo.Spike(0 * pq.s)
        unit.spikes.append(s)
        segment.spikes.append(s)
        s.unit = unit
        s.segment = segment

        st = neo.SpikeTrain([] * pq.s, 0 * pq.s)
        unit.spiketrains.append(st)
        segment.spiketrains.append(st)
        st.unit = unit
        st.segment = segment

        tools.remove_from_hierarchy(st)
        self.assertTrue(s in unit.spikes)
        self.assertTrue(s in segment.spikes)
        self.assertFalse(st in unit.spiketrains)
        self.assertFalse(st in segment.spiketrains)

    def test_extract_spikes(self):
        s1 = sp.zeros(10000)
        s2 = sp.ones(10000)
        t = sp.arange(0.0, 10.1, 1.0)

        sig1 = neo.AnalogSignal(s1 * pq.uV, sampling_rate=pq.kHz)
        sig2 = neo.AnalogSignal(s2 * pq.uV, sampling_rate=pq.kHz)
        train = neo.SpikeTrain(t * pq.s, 10 * pq.s)

        spikes = tools.extract_spikes(
            train, [sig1, sig2], 100 * pq.ms, 10 * pq.ms)

        self.assertEqual(len(spikes), 9)
        for s in spikes:
            self.assertAlmostEqual(s.waveform[:, 0].mean(), 0.0)
            self.assertAlmostEqual(s.waveform[:, 1].mean(), 1.0)

    def test_extract_different_spikes(self):
        s1 = sp.ones(10500)
        s2 = -sp.ones(10500)
        for i in xrange(10):
            s1[i * 1000 + 500:i * 1000 + 1500] *= i
            s2[i * 1000 + 500:i * 1000 + 1500] *= i
        t = sp.arange(0.0, 10.1, 1.0)

        sig1 = neo.AnalogSignal(s1 * pq.uV, sampling_rate=pq.kHz)
        sig2 = neo.AnalogSignal(s2 * pq.uV, sampling_rate=pq.kHz)
        train = neo.SpikeTrain(t * pq.s, 10 * pq.s)

        spikes = tools.extract_spikes(
            train, [sig1, sig2], 100 * pq.ms, 10 * pq.ms)

        self.assertEqual(len(spikes), 10)
        for i, s in enumerate(spikes):
            self.assertAlmostEqual(s.waveform[:, 0].mean(), i)
            self.assertAlmostEqual(s.waveform[:, 1].mean(), -i)

if __name__ == '__main__':
    ut.main()
