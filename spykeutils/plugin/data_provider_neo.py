import os
import sys
from copy import copy
from collections import OrderedDict
import traceback
import atexit

import neo

from data_provider import DataProvider
from .. import conversions as convert


class NeoDataProvider(DataProvider):
    """ Base class for data providers using NEO"""

    # Dictionary of block lists, indexed by (filename, block index) tuples
    loaded_blocks = {}
    # Dictionary of index in file, indexed by block object
    block_indices = {}
    # Dictionary of io, indexed by block object
    block_ios = {}
    # Dictionary of io (IO name, read paramters) tuples for loaded blocks
    block_read_params = {}
    # Mode for data lazy loading:
    # 0 - Full load
    # 1 - Lazy load
    # 2 - Caching lazy load
    data_lazy_mode = 0
    # Mode for lazy cascade
    cascade_lazy = False
    # Forced IO class for all files. If None, determine by file extension.
    forced_io = None
    # Active IO read parameters (dictionary indexed by IO class)
    io_params = {}

    def __init__(self, name, progress):
        super(NeoDataProvider, self).__init__(name, progress)

    @classmethod
    def clear(cls):
        """ Clears cached blocks
        """
        cls.loaded_blocks.clear()
        cls.block_indices.clear()
        cls.block_read_params.clear()

        ios = set()
        for io in cls.block_ios.itervalues():
            if io in ios:
                continue
            if hasattr(io, 'close'):
                io.close()
                ios.add(io)
        cls.block_ios.clear()

    @classmethod
    def get_block(cls, filename, index, lazy=None, force_io=None,
                  read_params=None):
        """ Return the block at the given index in the specified file.

        :param str filename: Path to the file from which to load the block.
        :param int index: The index of the block in the file.
        :param int lazy: Override global lazy setting if not ``None``:
            0 regular load, 1 lazy load, 2 caching lazy load.
        :param force_io: Override global forced_io for the Neo IO class
            to use when loading the file. If ``None``, the global
            forced_io is used.
        :param dict read_params: Override read parameters for the IO that
            will load the block. If ``None``, the global io_params are
            used.
        """
        if lazy is None:
            lazy = cls.data_lazy_mode > 0
        else:
            lazy = lazy > 0
        if force_io is None:
            force_io = cls.forced_io

        if filename in cls.loaded_blocks:
            return cls.loaded_blocks[filename][index]
        io, blocks = cls._load_neo_file(filename, lazy, force_io, read_params)
        if io and not lazy and not cls.cascade_lazy and hasattr(io, 'close'):
            io.close()
        if blocks is None:
            return None
        return blocks[index]

    @classmethod
    def get_blocks(cls, filename, lazy=None, force_io=None,
                   read_params=None):
        """ Return a list of blocks loaded from the specified file

        :param str filename: Path to the file from which to load the blocks.
        :param int lazy: Override global lazy setting if not ``None``:
            0 regular load, 1 lazy load, 2 caching lazy load.
        :param force_io: Override global forced_io for the Neo IO class
            to use when loading the file. If ``None``, the global
            forced_io is used.
        :param dict read_params: Override read parameters for the IO that
            will load the block. If ``None``, the global io_params are
            used.
        """
        if lazy is None:
            lazy = cls.data_lazy_mode > 0
        else:
            lazy = lazy > 0
        if force_io is None:
            force_io = cls.forced_io

        if filename in cls.loaded_blocks:
            return cls.loaded_blocks[filename]
        io, blocks = cls._load_neo_file(filename, lazy, force_io, read_params)
        if io and not lazy and not cls.cascade_lazy and hasattr(io, 'close'):
            io.close()
        return blocks

    @classmethod
    def _load_neo_file(cls, filename, lazy, force_io, read_params):
        """ Returns a NEO io object and a list of contained blocks for a
        file name. This function also caches all loaded blocks

        :param str filename: The full path of the file (relative or absolute).
        :param bool lazy: Determines if lazy mode is used for Neo io.
        :param force_io: IO class to use for loading. If None, determined
            by file extension or through trial and error for directories.
        :param dict read_params: Override read parameters for the IO that
            will load the block. If ``None``, the global io_params are
            used.
        """
        cascade = 'lazy' if cls.cascade_lazy else True
        if os.path.isdir(filename):
            if force_io:
                try:
                    n_io = force_io(filename)
                    if read_params is None:
                        rp = cls.io_params.get(force_io, {})
                    else:
                        rp = read_params
                    content = n_io.read(lazy=lazy, cascade=cascade, **rp)
                    if force_io == neo.TdtIO and \
                            isinstance(content, neo.Block) and \
                            not content.segments:
                        # TdtIO can produce empty blocks for invalid dirs
                        sys.stderr.write(
                            'Could not load any blocks from "%s"' % filename)
                        return None, None

                    return cls._content_loaded(
                        content, filename, lazy, n_io, rp)
                except Exception, e:
                    sys.stderr.write(
                        'Load error for directory "%s":\n' % filename)
                    tb = sys.exc_info()[2]
                    while not ('self' in tb.tb_frame.f_locals and
                               tb.tb_frame.f_locals['self'] == n_io):
                        if tb.tb_next is not None:
                            tb = tb.tb_next
                        else:
                            break
                    traceback.print_exception(type(e), e, tb)
            else:
                for io in neo.io.iolist:
                    if io.mode == 'dir':
                        try:
                            n_io = io(filename)
                            if read_params is None:
                                rp = cls.io_params.get(force_io, {})
                            else:
                                rp = read_params
                            content = n_io.read(lazy=lazy, cascade=cascade, **rp)
                            if io == neo.TdtIO and \
                                    isinstance(content, neo.Block) and \
                                    not content.segments:
                                # TdtIO can produce empty blocks for invalid dirs
                                continue

                            return cls._content_loaded(
                                content, filename, lazy, n_io, rp)
                        except Exception, e:
                            sys.stderr.write(
                                'Load error for directory "%s":\n' % filename)
                            tb = sys.exc_info()[2]
                            while not ('self' in tb.tb_frame.f_locals and
                                       tb.tb_frame.f_locals['self'] == n_io):
                                if tb.tb_next is not None:
                                    tb = tb.tb_next
                                else:
                                    break
                            traceback.print_exception(type(e), e, tb)
        else:
            if force_io:
                if read_params is None:
                    rp = cls.io_params.get(force_io, {})
                else:
                    rp = read_params
                return cls._load_file_with_io(filename, force_io, lazy, rp)

            extension = filename.split('.')[-1]
            for io in neo.io.iolist:
                if extension in io.extensions:
                    if read_params is None:
                        rp = cls.io_params.get(io, {})
                    else:
                        rp = read_params
                    return cls._load_file_with_io(filename, io, lazy, rp)

        return None, None

    @classmethod
    def _content_loaded(cls, content, filename, lazy, n_io, read_params):
        if isinstance(content, neo.Block):  # Neo 0.2.1
            cls.block_indices[content] = 0
            cls.loaded_blocks[filename] = [content]
            cls.block_read_params[content] = (type(n_io).__name__, read_params)
            if lazy or cls.cascade_lazy:
                cls.block_ios[content] = n_io
            return n_io, [content]

        # Neo >= 0.3.0, read() returns a list of blocks
        blocks = content
        for i, b in enumerate(blocks):
            cls.block_indices[b] = i
            cls.block_read_params[b] = (type(n_io).__name__, read_params)
            if lazy or cls.cascade_lazy:
                cls.block_ios[b] = n_io

        cls.loaded_blocks[filename] = blocks
        return n_io, blocks

    @classmethod
    def _load_file_with_io(cls, filename, io, lazy, read_params):
        if io == neo.NeoHdf5IO:
            # Fix unicode problem with pyinstaller
            if hasattr(sys, 'frozen'):
                filename = filename.encode('UTF-8')

        n_io = io(filename=filename)

        if read_params is None:
            rp = cls.io_params.get(io, {})
        else:
            rp = read_params

        try:
            cascade = 'lazy' if cls.cascade_lazy else True
            if hasattr(io, 'read_all_blocks'):  # Neo 0.2.1
                content = n_io.read_all_blocks(lazy=lazy, cascade=cascade, **rp)
            else:
                content = n_io.read(lazy=lazy, cascade=cascade, **rp)

            return cls._content_loaded(content, filename, lazy, n_io, rp)
        except Exception, e:
            sys.stderr.write(
                'Load error for file "%s":\n' % filename)
            tb = sys.exc_info()[2]
            while not ('self' in tb.tb_frame.f_locals and
                               tb.tb_frame.f_locals['self'] == n_io):
                if tb.tb_next is not None:
                    tb = tb.tb_next
                else:
                    break
            traceback.print_exception(type(e), e, tb)

        return None, None

    @classmethod
    def _get_data_from_viewer(cls, viewer):
        """ Return a dictionary with selection information from viewer
        """
        # The links in this data format are based list indices
        data = {}
        data['type'] = 'Neo'

        # Block entry: (Index of block in file, file location of block,
        # block IO class name, block IO read parameters)
        block_list = []
        block_indices = {}
        selected_blocks = viewer.neo_blocks()
        block_files = viewer.neo_block_file_names()
        for b in selected_blocks:
            block_indices[b] = len(block_list)
            block_list.append([NeoDataProvider.block_indices[b],
                               block_files[b],
                               cls.block_read_params[b][0],
                               cls.block_read_params[b][1]])
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

        # Unit entry: (Index of unit in rcg, index of rcg)
        unit_list = []
        selected_units = viewer.neo_units()
        for u in selected_units:
            segment_indices[u] = len(segment_list)
            rcg_id = None if u.recordingchannelgroup is None \
                else u.recordingchannelgroup.units.index(u)
            rcg = rcg_indices[u.recordingchannelgroup] \
                if u.recordingchannelgroup else None
            unit_list.append([rcg_id, rcg])
        data['units'] = unit_list

        return data

    @staticmethod
    def find_io_class(name):
        """ Return the Neo IO class with a given name.

        :param str name: Class name of the desired IO class.
        """
        for io in neo.io.iolist:
            if io.__name__ == name:
                return io
        return None

    def _active_block(self, old):
        """ Return a copy of all selected elements in the given block.
        Only container objects are copied, data objects are linked.

        Needs to load all lazily loaded objects and will cache them
        regardless of current lazy_mode,
        """
        block = copy(old)

        block.segments = []
        selected_segments = set(self.segments() + [None])
        selected_rcgs = set(self.recording_channel_groups() + [None])
        selected_channels = set(self.recording_channels() + [None])
        selected_units = set(self.units() + [None])

        for s in old.segments:
            if s in selected_segments:
                segment = copy(s)
                segment.analogsignals = [self._load_lazy_object(sig, True)
                                         for sig in s.analogsignals
                                         if sig.recordingchannel
                                         in selected_channels]
                segment.analogsignalarrays = [
                    self._load_lazy_object(asa, True)
                    for asa in s.analogsignalarrays
                    if asa.recordingchannelgroup in selected_rcgs]
                segment.irregularlysampledsignals = [
                    self._load_lazy_object(iss, True)
                    for iss in s.irregularlysampledsignals
                    if iss.recordingchannel in selected_channels]
                segment.spikes = [self._load_lazy_object(sp, True)
                                  for sp in s.spikes
                                  if sp.unit in selected_units]
                segment.spiketrains = [self._load_lazy_object(st, True)
                                       for st in s.spiketrains
                                       if st.unit in selected_units]
                segment.block = block
                block.segments.append(segment)

        block.recordingchannelgroups = []
        for old_rcg in old.recordingchannelgroups:
            if old_rcg in selected_rcgs:
                rcg = copy(old_rcg)
                rcg.analogsignalarrays = [
                    self._load_lazy_object(asa, True)
                    for asa in old_rcg.analogsignalarrays
                    if asa.segment in selected_segments]

                rcg.recordingchannels = []
                for c in old_rcg.recordingchannels:
                    if not c in selected_channels:
                        continue
                    channel = copy(c)
                    channel.analogsignals = [
                        self._load_lazy_object(sig, True)
                        for sig in c.analogsignals
                        if sig.segment in selected_segments]
                    channel.irregularlysampledsignals = [
                        self._load_lazy_object(iss, True)
                        for iss in c.irregularlysampledsignals
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
                    unit.spikes = [self._load_lazy_object(sp, True)
                                   for sp in u.spikes
                                   if sp.segment in selected_segments]
                    unit.spiketrains = [self._load_lazy_object(st, True)
                                        for st in u.spiketrains
                                        if st.segment in selected_segments]
                    unit.recordingchannelgroup = rcg
                    rcg.units.append(unit)

                rcg.block = block
                block.recordingchannelgroups.append(rcg)

        return block

    def _get_object_io(self, o):
        """ Find the IO for an object. Return ``None`` if no IO exists.
        """
        if o.segment:
            return self.block_ios.get(o.segment.block, None)
        if hasattr(object, 'recordingchannelgroups'):
            if o.recordingchannelgroups:
                return self.block_ios.get(
                    o.recordingchannelgroups[0].block, None)
        if hasattr(object, 'recordingchannel'):
            c = o.recordingchannel
            if c.recordingchannelgroups:
                return self.block_ios.get(
                    c.recordingchannelgroups[0].block, None)
        return None

    def _load_lazy_object(self, o, change_links=False):
        """ Return a loaded version of a lazily loaded object. The IO
        needs a ``read_lazy_object`` that takes a lazily loaded data object
        as parameter method for this to work.

        :param o: The object to load.
        :param bool change_links: If ``True``, replace the old object
            in the hierarchy.
        """
        if not hasattr(o, 'lazy_shape'):
            return o

        io = self._get_object_io(o)

        if io:
            if hasattr(io, 'load_lazy_object'):
                ret = io.load_lazy_object(o)
            elif isinstance(io, neo.io.NeoHdf5IO):
                ret = io.get(o.hdf5_path, cascade=False, lazy=False)
            else:
                return o

            ret.segment = o.segment
            if hasattr(o, 'recordingchannelgroup'):
                ret.recordingchannelgroup = o.recordingchannelgroup
            elif hasattr(o, 'recordingchannel'):
                ret.recordingchannel = o.recordingchannel
            elif hasattr(o, 'unit'):
                ret.unit = o.unit

            if change_links:
                name = type(o).__name__.lower() + 's'
                l = getattr(o.segment, name)
                try:
                    l[l.index(o)] = ret
                except ValueError:
                    l.append(ret)

                l = None
                if hasattr(o, 'recordingchannelgroup'):
                    l = getattr(o.recordingchannelgroup, name)
                elif hasattr(o, 'recordingchannel'):
                    l = getattr(o.recordingchannel, name)
                elif hasattr(o, 'unit'):
                    l = getattr(o.unit, name)
                if l is not None:
                    try:
                        l[l.index(o)] = ret
                    except ValueError:
                        l.append(ret)

            return ret
        return o

    def _load_object_list(self, objects):
        """ Return a list of loaded objects for a list of (potentially)
        lazily loaded objects.
        """
        ret = []
        for o in objects:
            ret.append(self._load_lazy_object(o, self.data_lazy_mode > 1))
        return ret

    def _load_object_dict(self, objects):
        """ Return a dictionary (without changing indices) of loaded
        objects for a dictionary of (potentially) lazily loaded objects.
        """
        for k, v in objects.items():
            if isinstance(v, list):
                objects[k] = self._load_object_list(v)
            elif isinstance(v, dict):
                for ik, iv in v.items():
                    v[ik] = self._load_lazy_object(iv, self.data_lazy_mode > 1)
            else:
                raise ValueError(
                    'Only dicts or lists are supported as dictionary values!')
        return objects

    def selection_blocks(self):
        """ Return a list of selected blocks.
        """
        return [self._active_block(b) for b in self.blocks()]

    def spike_trains(self):
        """ Return a list of :class:`neo.core.SpikeTrain` objects.
        """
        trains = []
        units = set(self.units())
        for s in self.segments():
            trains.extend([t for t in s.spiketrains if t.unit in units or
                           t.unit is None])
        for u in self.units():
            trains.extend([t for t in u.spiketrains if t.segment is None])

        return self._load_object_list(trains)

    def spike_trains_by_unit(self):
        """ Return a dictionary (indexed by Unit) of lists of
        :class:`neo.core.SpikeTrain` objects.
        """
        trains = OrderedDict()
        segments = set(self.segments())
        for u in self.units():
            st = [t for t in u.spiketrains if t.segment in segments or
                  t.segment is None]
            if st:
                trains[u] = st

        nonetrains = []
        for s in self.segments():
            nonetrains.extend([t for t in s.spiketrains if t.unit is None])
        if nonetrains:
            trains[self.no_unit] = nonetrains

        return self._load_object_dict(trains)

    def spike_trains_by_segment(self):
        """ Return a dictionary (indexed by Segment) of lists of
        :class:`neo.core.SpikeTrain` objects.
        """
        trains = OrderedDict()
        units = self.units()
        for s in self.segments():
            st = [t for t in s.spiketrains if t.unit in units or
                  t.unit is None]
            if st:
                trains[s] = st

        nonetrains = []
        for u in self.units():
            nonetrains.extend([t for t in u.spiketrains if t.segment is None])
        if nonetrains:
            trains[self.no_segment] = nonetrains

        return self._load_object_dict(trains)

    def spike_trains_by_unit_and_segment(self):
        """ Return a dictionary (indexed by Unit) of dictionaries
        (indexed by Segment) of :class:`neo.core.SpikeTrain` objects.
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

        return self._load_object_dict(trains)

    def spikes(self):
        """ Return a list of :class:`neo.core.Spike` objects.
        """
        spikes = []
        units = self.units()
        for s in self.segments():
            spikes.extend([t for t in s.spikes if t.unit in units or
                           t.unit is None])
        for u in self.units():
            spikes.extend([t for t in u.spikes if t.segment is None])

        return self._load_object_list(spikes)

    def spikes_by_unit(self):
        """ Return a dictionary (indexed by Unit) of lists of
        :class:`neo.core.Spike` objects.
        """
        spikes = OrderedDict()
        segments = self.segments()
        for u in self.units():
            sp = [t for t in u.spikes if t.segment in segments or
                  t.segment is None]
            if sp:
                spikes[u] = sp

        nonespikes = []
        for s in self.segments():
            nonespikes.extend([t for t in s.spikes if t.unit is None])
        if nonespikes:
            spikes[self.no_unit] = nonespikes

        return self._load_object_dict(spikes)

    def spikes_by_segment(self):
        """ Return a dictionary (indexed by Segment) of lists of
        :class:`neo.core.Spike` objects.
        """
        spikes = OrderedDict()
        units = self.units()
        for s in self.segments():
            sp = [t for t in s.spikes if t.unit in units or
                  t.unit is None]
            if sp:
                spikes[s] = sp

        nonespikes = []
        for u in self.units():
            nonespikes.extend([t for t in u.spikes if t.segment is None])
        if nonespikes:
            spikes[self.no_segment] = nonespikes

        return self._load_object_dict(spikes)

    def spikes_by_unit_and_segment(self):
        """ Return a dictionary (indexed by Unit) of dictionaries
        (indexed by Segment) of :class:`neo.core.Spike` lists.
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

        return self._load_object_dict(spikes)

    def events(self, include_array_events=True):
        """ Return a dictionary (indexed by Segment) of lists of
        Event objects.
        """
        ret = OrderedDict()
        for s in self.segments():
            if s.events:
                ret[s] = s.events
            if include_array_events:
                for a in s.eventarrays:
                    if s not in ret:
                        ret[s] = []
                    ret[s].extend(convert.event_array_to_events(a))
        return ret

    def labeled_events(self, label, include_array_events=True):
        """ Return a dictionary (indexed by Segment) of lists of Event
        objects with the given label.
        """
        ret = OrderedDict()
        for s in self.segments():
            events = [e for e in s.events if e.label == label]
            if events:
                ret[s] = events
            if include_array_events:
                for a in s.eventarrays:
                    if s not in ret:
                        ret[s] = []
                    events = convert.event_array_to_events(a)
                    ret[s].extend((e for e in events if e.label == label))
        return ret

    def event_arrays(self):
        """ Return a dictionary (indexed by Segment) of lists of
        EventArray objects.
        """
        ret = OrderedDict()
        for s in self.segments():
            if s.eventarrays:
                ret[s] = s.eventarrays
        return self._load_object_dict(ret)

    def epochs(self, include_array_epochs=True):
        """ Return a dictionary (indexed by Segment) of lists of
        Epoch objects.
        """
        ret = OrderedDict()
        for s in self.segments():
            if s.epochs:
                ret[s] = s.epochs
            if include_array_epochs:
                for a in s.epocharrays:
                    if s not in ret:
                        ret[s] = []
                    ret[s].extend(convert.epoch_array_to_epochs(a))
        return ret

    def labeled_epochs(self, label, include_array_epochs=True):
        """ Return a dictionary (indexed by Segment) of lists of Epoch
        objects with the given label.
        """
        ret = OrderedDict()
        for s in self.segments():
            epochs = [e for e in s.epochs if e.label == label]
            if epochs:
                ret[s] = epochs
            if include_array_epochs:
                for a in s.epocharrays:
                    if s not in ret:
                        ret[s] = []
                    epochs = convert.epoch_array_to_epochs(a)
                    ret[s].extend((e for e in epochs if e.label == label))
        return ret

    def epoch_arrays(self):
        """ Return a dictionary (indexed by Segment) of lists of
        EpochArray objects.
        """
        ret = OrderedDict()
        for s in self.segments():
            if s.epocharrays:
                ret[s] = s.epocharrays
        return self._load_object_dict(ret)

    def analog_signals(self, conversion_mode=1):
        """ Return a list of :class:`neo.core.AnalogSignal` objects.
        """
        signals = []
        channels = self.recording_channels()

        if conversion_mode == 1 or conversion_mode == 3:
            for s in self.segments():
                signals.extend([t for t in s.analogsignals
                                if t.recordingchannel in channels or
                                t.recordingchannel is None])
            for u in self.recording_channels():
                signals.extend([t for t in u.analogsignals
                                if t.segment is None])
        if conversion_mode > 1:
            for sa in self.analog_signal_arrays():
                for sig in convert.analog_signal_array_to_analog_signals(sa):
                    if (sig.recordingchannel is None or
                            sig.recordingchannel in channels):
                        signals.append(sig)

        return self._load_object_list(signals)

    def analog_signals_by_segment(self, conversion_mode=1):
        """ Return a dictionary (indexed by Segment) of lists of
        :class:`neo.core.AnalogSignal` objects.
        """
        signals = OrderedDict()
        channels = self.recording_channels()

        if conversion_mode == 1 or conversion_mode == 3:
            for s in self.segments():
                sig = []
                for c in channels:
                    sig.extend([t for t in c.analogsignals
                                if t.segment == s])
                sig.extend([t for t in s.analogsignals
                            if t.recordingchannel is None])
                if sig:
                    signals[s] = sig

            nonesignals = []
            for c in channels:
                nonesignals.extend([t for t in c.analogsignals
                                    if t.segment is None])
            if nonesignals:
                signals[self.no_segment] = nonesignals

        if conversion_mode > 1:
            for o, sa_list in \
                    self.analog_signal_arrays_by_segment().iteritems():
                for sa in sa_list:
                    for sig in \
                            convert.analog_signal_array_to_analog_signals(sa):
                        if sig.recordingchannel is None or \
                                sig.recordingchannel in channels:
                            if o not in signals:
                                signals[o] = []
                            signals[o].append(sig)

        return self._load_object_dict(signals)

    def analog_signals_by_channel(self, conversion_mode=1):
        """ Return a dictionary (indexed by RecordingChannel) of lists
        of :class:`neo.core.AnalogSignal` objects.
        """
        signals = OrderedDict()
        channels = self.recording_channels()

        if conversion_mode == 1 or conversion_mode == 3:
            segments = self.segments()
            for c in channels:
                sig = [t for t in c.analogsignals
                       if t.segment in segments or
                       t.segment is None]
                if sig:
                    signals[c] = sig

            nonesignals = []
            for s in segments:
                nonesignals.extend([t for t in s.analogsignals
                                    if t.recordingchannel is None])
            if nonesignals:
                signals[self.no_channel] = nonesignals

        if conversion_mode > 1:
            for o, sa_list in \
                    self.analog_signal_arrays_by_channelgroup().iteritems():
                for sa in sa_list:
                    for sig in \
                            convert.analog_signal_array_to_analog_signals(sa):
                        if sig.recordingchannel is None:
                            if self.no_channel not in signals:
                                signals[self.no_channel] = [sig]
                            else:
                                signals[self.no_channel].append(sig)
                        elif sig.recordingchannel in channels:
                            if sig.recordingchannel not in signals:
                                signals[sig.recordingchannel] = [sig]
                            else:
                                signals[sig.recordingchannel].append(sig)

        return self._load_object_dict(signals)

    def analog_signals_by_channel_and_segment(self, conversion_mode=1):
        """ Return a dictionary (indexed by RecordingChannel) of
        dictionaries (indexed by Segment) of :class:`neo.core.AnalogSignal`
        lists.
        """
        signals = OrderedDict()
        channels = self.recording_channels()

        if conversion_mode == 1 or conversion_mode == 3:
            segments = self.segments()
            for c in channels:
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

        if conversion_mode > 1:
            sigs = self.analog_signal_arrays_by_channelgroup_and_segment()
            for cg, inner in sigs.iteritems():
                for seg, sa_list in inner.iteritems():
                    for sa in sa_list:
                        for sig in convert.analog_signal_array_to_analog_signals(sa):
                            chan = sig.recordingchannel
                            if chan not in channels:
                                continue
                            if chan not in signals:
                                signals[chan] = OrderedDict()
                            if seg not in signals[chan]:
                                signals[chan][seg] = []
                            signals[chan][seg].append(sig)

        return self._load_object_dict(signals)

    def analog_signal_arrays(self):
        """ Return a list of :class:`neo.core.AnalogSignalArray` objects.
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

        return self._load_object_list(signals)

    def analog_signal_arrays_by_segment(self):
        """ Return a dictionary (indexed by Segment) of lists of
        :class:`neo.core.AnalogSignalArray` objects.
        """
        signals = OrderedDict()
        channelgroups = self.recording_channel_groups()
        for s in self.segments():
            sa = []
            for c in channelgroups:
                sa.extend([t for t in c.analogsignalarrays
                           if t.segment == s])
            sa.extend([t for t in s.analogsignalarrays
                       if t.recordingchannelgroup is None])
            if sa:
                signals[s] = sa

        nonesignals = []
        for c in channelgroups:
            nonesignals.extend([t for t in c.analogsignalarrays
                                if t.segment is None])
        if nonesignals:
            signals[self.no_segment] = nonesignals

        return self._load_object_dict(signals)

    def analog_signal_arrays_by_channelgroup(self):
        """ Return a dictionary (indexed by RecordingChannelGroup) of
        lists of :class:`neo.core.AnalogSignalArray` objects.
        """
        signals = OrderedDict()
        segments = self.segments()
        for c in self.recording_channel_groups():
            sa = [t for t in c.analogsignalarrays
                  if t.segment in segments]
            if sa:
                signals[c] = sa

        nonesignals = []
        for s in segments:
            nonesignals.extend([t for t in s.analogsignalarrays
                                if t.recordingchannelgroup is None])
        if nonesignals:
            signals[self.no_channelgroup] = nonesignals

        return self._load_object_dict(signals)

    def analog_signal_arrays_by_channelgroup_and_segment(self):
        """ Return a dictionary (indexed by RecordingChannelGroup) of
        dictionaries (indexed by Segment) of
        :class:`neo.core.AnalogSignalArray` lists.
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
                    signals[c][s] = segsignals
            nonesignals = [t for t in c.analogsignalarrays
                           if t.segment is None]
            if nonesignals:
                if c not in signals:
                    signals[c] = OrderedDict()
                signals[c][self.no_segment] = nonesignals

        nonesignals = OrderedDict()
        for s in self.segments():
            segsignals = [t for t in s.analogsignalarrays
                          if t.recordingchannelgroup is None]
            if segsignals:
                nonesignals[s] = segsignals
        if nonesignals:
            signals[self.no_channelgroup] = nonesignals

        return self._load_object_dict(signals)


atexit.register(NeoDataProvider.clear)