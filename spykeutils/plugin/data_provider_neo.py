import os
import sys
from copy import copy
from collections import OrderedDict

import neo

from data_provider import DataProvider
from .. import conversions as convert


class NeoDataProvider(DataProvider):
    """ Base class for data providers using NEO"""

    # Dictionary of block lists, indexed by (filename, block index) tuples
    loaded_blocks = {}
    # Dictionary of index in file, indexed by block object
    block_indices = {}

    def __init__(self, name, progress):
        super(NeoDataProvider, self).__init__(name, progress)

    @classmethod
    def clear(cls):
        """ Clears cached blocks
        """
        cls.loaded_blocks.clear()
        cls.block_indices.clear()

    @classmethod
    def get_block(cls, filename, index, lazy=False):
        """ Return the block at the given index in the specified file
        """
        if filename in cls.loaded_blocks:
            return cls.loaded_blocks[filename][index]
        io, blocks = cls._load_neo_file(filename, lazy)
        if io and hasattr(io, 'close'):
            io.close()
        return blocks[index]

    @classmethod
    def get_blocks(cls, filename, lazy=False):
        """ Return a list of blocks loaded from the specified file
        """
        if filename in cls.loaded_blocks:
            return cls.loaded_blocks[filename]
        io, blocks = cls._load_neo_file(filename, lazy)
        if io and hasattr(io, 'close'):
            io.close()
        return blocks

    @classmethod
    def _load_neo_file(cls, filename, lazy):
        """ Returns a NEO io object and a list of contained blocks for a
            file name. This function also caches all loaded blocks
            :Parameters:
                filename : str
                    The full path of the file (relative or absolute)
                lazy : bool
                    Determines if lazy mode is used for NEO io
        """
        if os.path.isdir(filename):
            for io in neo.io.iolist:
                if io.mode == 'dir':
                    n_io = io(filename)
                    block = n_io.read(lazy=lazy)
                    cls.block_indices[block] = 0
                    cls.loaded_blocks[filename] = [block]
                    return n_io, [block]
        else:
            extension = filename.split('.')[-1]
            for io in neo.io.iolist:
                if extension in io.extensions:
                    if io == neo.NeoHdf5IO:
                        # Fix unicode problem with pyinstaller
                        if hasattr(sys, 'frozen'):
                            filename = filename.encode('UTF-8')
                        n_io = io(filename=filename)
                        blocks = n_io.read_all_blocks(lazy=lazy)
                        for i, b in enumerate(blocks):
                            cls.block_indices[b] = i
                        cls.loaded_blocks[filename] = blocks
                        return n_io, blocks
                    try:
                        n_io = io(filename=filename)
                        block = n_io.read(lazy=lazy)
                        cls.block_indices[block] = 0
                        cls.loaded_blocks[filename] = [block]
                        return n_io, [block]
                    except Exception:
                        sys.stderr.write('Load error with '+str(io)+
                                         ' for file '+filename+'\n')
                        continue
        return None, None

    @staticmethod
    def _get_data_from_viewer(viewer):
        """ Return a dictionary with selection information from viewer
        """
        # The links in this data format are based list indices
        data = {}
        data['type'] = 'Neo'

        # Block entry: (Index of block in file, file location of block)
        block_list = []
        block_indices = {}
        selected_blocks = viewer.neo_blocks()
        block_files = viewer.neo_block_file_names()
        for b in selected_blocks:
            block_indices[b] = len(block_list)
            block_list.append([NeoDataProvider.block_indices[b],
                               block_files[b]])
        data['blocks'] = block_list

        # Recording channel group entry:
        # (Index of rcg in block, index of block)
        rcg_list = []
        rcg_indices = {}
        selected_rcg = viewer.neo_channel_groups()
        for rcg in selected_rcg:
            rcg_indices[rcg] = len(rcg_list)
            idx = rcg.block.recordingchannelgroups.index(rcg)
            rcg_list.append([idx, block_indices[rcg.block]])
        data['channel_groups'] = rcg_list

        # Recording channel entry: (Index of channel in rcg, index of rcg)
        # There can be multiple channel entries for one channel object, if
        # it is part of multiple channel groups
        channel_list = []
        selected_channels = viewer.neo_channels()
        for c in selected_channels:
            for rcg in c.recordingchannelgroups:
                if rcg in rcg_indices:
                    idx = rcg.recordingchannels.index(c)
                    channel_list.append([idx, rcg_indices[rcg]])
        data['channels'] = channel_list

        # Segment entry: (Index of segment in block, index of block)
        segment_list = []
        segment_indices = {}
        selected_segments = viewer.neo_segments()
        for s in selected_segments:
            segment_indices[s] = len(segment_list)
            idx = s.block.segments.index(s)
            segment_list.append([idx, block_indices[s.block]])
        data['segments'] = segment_list

        # Unit entry: (Index of uinit in rcg, index of rcg)
        unit_list = []
        selected_units = viewer.neo_units()
        for u in selected_units:
            segment_indices[u] = len(segment_list)
            rcg_id = None if u.recordingchannelgroup is None\
            else u.recordingchannelgroup.units.index(u)
            rcg = rcg_indices[u.recordingchannelgroup]\
            if u.recordingchannelgroup else None
            unit_list.append([rcg_id, rcg])
        data['units'] = unit_list

        return data

    def spike_trains(self):
        """ Return a list of SpikeTrain objects.
        """
        trains = []
        units = self.units()
        for s in self.segments():
            trains.extend([t for t in s.spiketrains if t.unit in units or
                                                       t.unit is None])
        for u in self.units():
            trains.extend([t for t in u.spiketrains if t.segment is None])

        return trains

    def spike_trains_by_unit(self):
        """ Return a dictionary (indexed by Unit) of lists of
        SpikeTrain objects.
        """
        trains = OrderedDict()
        segments = self.segments()
        for u in self.units():
            trains[u] = [t for t in u.spiketrains if t.segment in segments]

        nonetrains = []
        for s in self.segments():
            nonetrains.extend([t for t in s.spiketrains if t.unit is None])
        if nonetrains:
            trains[self.no_unit] = nonetrains

        return trains

    def _active_block(self, old):
        """ Return a copy of all selected elements in the given block
        """
        block = copy(old)

        block.segments = []
        selected_segments = self.segments()
        selected_rcgs = self.recording_channel_groups()
        selected_channels = self.recording_channels()
        selected_units = self.units()
        for s in old.segments:
            if s in selected_segments:
                segment = copy(s)
                segment.analogsignals = [sig for sig in s.analogsignals
                                         if sig.recordingchannel
                                         in selected_channels]
                segment.analogsignalarrays = \
                    [asa for asa in s.analogsignalarrays
                     if asa.recordingchannelgroup in selected_rcgs]
                segment.irregularlysampledsignals = \
                    [iss for iss in s.irregularlysampledsignals
                     if iss.recordingchannel in selected_channels]
                segment.spikes = [sp for sp in s.spikes
                                  if sp.unit in selected_units]
                segment.spiketrains = [st for st in s.spiketrains
                                        if st.unit in selected_units]
                segment.block = block
                block.segments.append(segment)

        block.recordingchannelgroups = []
        for old_rcg in old.recordingchannelgroups:
            if old_rcg in selected_rcgs:
                rcg = copy(old_rcg)
                rcg.analogsignalarrays =\
                    [asa for asa in old_rcg.analogsignalarrays
                     if asa.segment in selected_segments]

                rcg.recordingchannels = []
                for c in old_rcg.recordingchannels:
                    if not c in selected_channels:
                        continue
                    channel = copy(c)
                    channel.analogsignals = [sig for sig in c.analogsignals
                                             if sig.segment
                                             in selected_segments]
                    channel.irregularlysampledsignals =\
                        [iss for iss in c.irregularlysampledsignals
                         if iss.segment in selected_segments]
                    channel.recordingchannelgroups = copy(
                        c.recordingchannelgroups)
                    channel.recordingchannelgroups.insert(
                        channel.recordingchannelgroups.index(old_rcg), rcg)
                    channel.recordingchannelgroups.remove(old_rcg)
                    rcg.recordingchannels.append(channel)

                rcg.units = []
                for u in old_rcg.units:
                    if not u in selected_units:
                        continue

                    unit = copy(u)
                    unit.spikes = [sp for sp in u.spikes
                                   if sp.segment in selected_segments]
                    unit.spiketrains = [st for st in u.spiketrains
                                         if st.segment in selected_segments]
                    unit.recordingchannelgroup = rcg
                    rcg.units.append(unit)

                rcg.block = block
                block.recordingchannelgroups.append(rcg)

        return block

    def selection_blocks(self):
        """ Return a list of selected blocks.
        """
        return [self._active_block(b) for b in self.blocks()]


    def spike_trains_by_segment(self):
        """ Return a dictionary (indexed by Segment) of lists of
        SpikeTrain objects.
        """
        trains = OrderedDict()
        units = self.units()
        for s in self.segments():
            trains[s] = [t for t in s.spiketrains if t.unit in units]

        nonetrains = []
        for u in self.units():
            nonetrains.extend([t for t in u.spiketrains if t.segment is None])
        if nonetrains:
            trains[self.no_segment] = nonetrains

        return trains

    def spike_trains_by_unit_and_segment(self):
        """ Return a dictionary (indexed by Unit) of dictionaries
        (indexed by Segment) of SpikeTrain objects.
        """
        trains = OrderedDict()
        segments = self.segments()
        for u in self.units():
            for s in segments:
                segtrains = [t for t in u.spiketrains if t.segment == s]
                if segtrains:
                    if u not in trains:
                        trains[u] = OrderedDict()
                    trains[u][s] = segtrains[0]
            nonetrains = [t for t in u.spiketrains if t.segment is None]
            if nonetrains:
                if u not in trains:
                    trains[u] = OrderedDict()
                trains[u][self.no_segment] = nonetrains[0]

        nonetrains = OrderedDict()
        for s in self.segments():
            segtrains = [t for t in s.spiketrains if t.unit is None]
            if segtrains:
                nonetrains[s] = segtrains[0]
        if nonetrains:
            trains[self.no_unit] = nonetrains

        return trains

    def spikes(self):
        """ Return a list of Spike objects.
        """
        spikes = []
        units = self.units()
        for s in self.segments():
            spikes.extend([t for t in s.spikes if t.unit in units or
                                                  t.unit is None])
        for u in self.units():
            spikes.extend([t for t in u.spikes if t.segment is None])

        return spikes

    def spikes_by_unit(self):
        """ Return a dictionary (indexed by Unit) of lists of
        Spike objects.
        """
        spikes = OrderedDict()
        segments = self.segments()
        for u in self.units():
            spikes[u] = [t for t in u.spikes if t.segment in segments]

        nonespikes = []
        for s in self.segments():
            nonespikes.extend([t for t in s.spikes if t.unit is None])
        if nonespikes:
            spikes[self.no_unit] = nonespikes

        return spikes

    def spikes_by_segment(self):
        """ Return a dictionary (indexed by Segment) of lists of
        Spike objects.
        """
        spikes = OrderedDict()
        units = self.units()
        for s in self.segments():
            spikes[s] = [t for t in s.spikes if t.unit in units]

        nonespikes = []
        for u in self.units():
            nonespikes.extend([t for t in u.spikes if t.segment is None])
        if nonespikes:
            spikes[self.no_segment] = nonespikes

        return spikes

    def spikes_by_unit_and_segment(self):
        """ Return a dictionary (indexed by Unit) of dictionaries
        (indexed by Segment) of Spike lists.
        """
        spikes = OrderedDict()
        segments = self.segments()
        for u in self.units():
            for s in segments:
                segtrains = [t for t in u.spikes if t.segment == s]
                if segtrains:
                    if u not in spikes:
                        spikes[u] = OrderedDict()
                    spikes[u][s] = segtrains
            nonespikes = [t for t in u.spikes if t.segment is None]
            if nonespikes:
                if u not in spikes:
                    spikes[u] = OrderedDict()
                spikes[u][self.no_segment] = nonespikes

        nonespikes = OrderedDict()
        for s in self.segments():
            segspikes = [t for t in s.spikes if t.unit is None]
            if segspikes:
                nonespikes[s] = segspikes
        if nonespikes:
            spikes[self.no_unit] = nonespikes

        return spikes

    def events(self, include_array_events = True):
        """ Return a dictionary (indexed by Segment) of lists of
        Event objects.
        """
        ret = OrderedDict()
        for s in self.segments():
            ret[s] = s.events
            if include_array_events:
                for a in s.eventarrays:
                    ret[s].extend(convert.event_array_to_events(a))
        return ret

    def labeled_events(self, label, include_array_events = True):
        """ Return a dictionary (indexed by Segment) of lists of Event
        objects with the given label.
        """
        ret = OrderedDict()
        for s in self.segments():
            ret[s] = [e for e in s.events if e.label == label]
            if include_array_events:
                for a in s.eventarrays:
                    events = convert.event_array_to_events(a)
                    ret[s].extend((e for e in events if e.label == label))
        return ret

    def event_arrays(self):
        """ Return a dictionary (indexed by Segment) of lists of
        EventArray objects.
        """
        ret = OrderedDict()
        for s in self.segments():
            ret[s] = s.eventarrays
        return ret

    def epochs(self, include_array_epochs = True):
        """ Return a dictionary (indexed by Segment) of lists of
        Epoch objects.
        """
        ret = OrderedDict()
        for s in self.segments():
            ret[s] = s.epochs
            if include_array_epochs:
                for a in s.epocharrays:
                    ret[s].extend(convert.epoch_array_to_epochs(a))
        return ret

    def labeled_epochs(self, label, include_array_epochs = True):
        """ Return a dictionary (indexed by Segment) of lists of Epoch
        objects with the given label.
        """
        ret = OrderedDict()
        for s in self.segments():
            ret[s] = [e for e in s.epochs if e.label == label]
            if include_array_epochs:
                for a in s.epocharrays:
                    epochs = convert.epoch_array_to_epochs(a)
                    ret[s].extend((e for e in epochs if e.label == label))
        return ret

    def epoch_arrays(self):
        """ Return a dictionary (indexed by Segment) of lists of
        EpochArray objects.
        """
        ret = OrderedDict()
        for s in self.segments():
            ret[s] = s.epocharrays
        return ret

    def num_analog_signals(self):
        """ Return the number of AnalogSignal objects.
        """
        return len(self.analog_signals())

    def analog_signals(self):
        """ Return a list of AnalogSignal objects.
        """
        signals = []
        channels = self.recording_channels()
        for s in self.segments():
            signals.extend([t for t in s.analogsignals
                           if t.recordingchannel in channels or
                              t.recordingchannel is None])
        for u in self.recording_channels():
            signals.extend([t for t in u.analogsignals if t.segment is None])

        return signals

    def analog_signals_by_segment(self):
        """ Return a dictionary (indexed by Segment) of lists of
        AnalogSignal objects.
        """
        signals = OrderedDict()
        channels = self.recording_channels()
        for s in self.segments():
            signals[s] = [t for t in s.analogsignals
                          if t.recordingchannel in channels or
                             t.recordingchannel is None]

        nonesignals = []
        for c in channels:
            nonesignals.extend([t for t in c.analogsignals
                                if t.segment is None])
        if nonesignals:
            signals[self.no_segment] = nonesignals

        return signals

    def analog_signals_by_channel(self):
        """ Return a dictionary (indexed by RecordingChannel) of lists
        of AnalogSignal objects.
        """
        signals = OrderedDict()
        segments = self.segments()
        for c in self.recording_channels():
            signals[c] = [t for t in c.analogsignals
                          if t.segment in segments or
                             t.segment is None]

        nonesignals = []
        for s in segments:
            nonesignals.extend([t for t in s.analogsignals
                                if t.recordingchannel is None])
        if nonesignals:
            signals[self.no_segment] = nonesignals

        return signals

    def analog_signals_by_channel_and_segment(self):
        """ Return a dictionary (indexed by RecordingChannel) of
        dictionaries (indexed by Segment) of AnalogSignal lists.
        """
        signals = OrderedDict()
        segments = self.segments()
        for c in self.recording_channels():
            for s in segments:
                segsignals = [t for t in c.analogsignals if t.segment == s]
                if segsignals:
                    if c not in signals:
                        signals[c] = OrderedDict()
                    signals[c][s] = segsignals
            nonesignals = [t for t in c.analogsignals if t.segment is None]
            if nonesignals:
                if c not in signals:
                    signals[c] = OrderedDict()
                signals[c][self.no_segment] = nonesignals

        nonesignals = OrderedDict()
        for s in self.segments():
            segsignals = [t for t in s.analogsignals
                          if t.recordingchannel is None]
            if segsignals:
                nonesignals[s] = segsignals
        if nonesignals:
            signals[self.no_channel] = nonesignals

        return signals

    def num_analog_signal_arrays(self):
        """ Return the number of AnalogSignalArray objects.
        """
        return len(self.analog_signal_arrays())

    def analog_signal_arrays(self):
        """ Return a list of AnalogSignalArray objects.
        """
        signals = []
        channelgroups = self.recording_channel_groups()
        for s in self.segments():
            signals.extend([t for t in s.analogsignalarrays
                            if t.recordingchannelgroup in channelgroups or
                            t.recordingchannelgroup is None])
        for u in channelgroups:
            signals.extend([t for t in u.analogsignalarrays
                            if t.segment is None])

        return signals

    def analog_signal_arrays_by_segment(self):
        """ Return a dictionary (indexed by Segment) of lists of
        AnalogSignalArray objects.
        """
        signals = OrderedDict()
        channelgroups = self.recording_channel_groups()
        for s in self.segments():
            signals[s] = [t for t in s.analogsignalarrays
                          if t.recordingchannelgroup in channelgroups]

        nonesignals = []
        for c in channelgroups:
            nonesignals.extend([t for t in c.analogsignalarrays
                                if t.segment is None])
        if nonesignals:
            signals[self.no_segment] = nonesignals

        return signals

    def analog_signal_arrays_by_channelgroup(self):
        """ Return a dictionary (indexed by RecordingChannelGroup) of
        lists of AnalogSignalArray objects.
        """
        signals = OrderedDict()
        segments = self.segments()
        for c in self.recording_channel_groups():
            signals[c] = [t for t in c.analogsignalarrays
                          if t.segment in segments]

        nonesignals = []
        for s in segments:
            nonesignals.extend([t for t in s.analogsignalarrays
                                if t.recordingchannelgroup is None])
        if nonesignals:
            signals[self.no_channelgroup] = nonesignals

        return signals

    def analog_signal_arrays_by_channelgroup_and_segment(self):
        """ Return a dictionary (indexed by RecordingChannelGroup) of
        dictionaries (indexed by Segment) of AnalogSignalArray objects.
        """
        signals = OrderedDict()
        segments = self.segments()
        for c in self.recording_channel_groups():
            for s in segments:
                segsignals = [t for t in c.analogsignalarrays
                              if t.segment == s]
                if segsignals:
                    if c not in signals:
                        signals[c] = OrderedDict()
                    signals[c][s] = segsignals[0]
            nonesignals = [t for t in c.analogsignalarrays
                           if t.segment is None]
            if nonesignals:
                if c not in signals:
                    signals[c] = OrderedDict()
                signals[c][self.no_segment] = nonesignals[0]

        nonesignals = OrderedDict()
        for s in self.segments():
            segsignals = [t for t in s.analogsignalarrays
                          if t.recordingchannelgroup is None]
            if segsignals:
                nonesignals[s] = segsignals[0]
        if nonesignals:
            signals[self.no_channelgroup] = nonesignals

        return signals