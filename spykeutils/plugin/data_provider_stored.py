import json
from data_provider import DataProvider
from data_provider_neo import NeoDataProvider
from ..progress_indicator import ProgressIndicator

class NeoStoredProvider(NeoDataProvider):
    def __init__(self, data, progress=ProgressIndicator()):
        super(NeoStoredProvider, self).__init__(data['name'], progress)
        self.data = data
        self.block_cache = None

    @classmethod
    def from_current_selection(cls, name, viewer):
        """ Create new NeoStoredProvider from current viewer selection
        """
        data = cls._get_data_from_viewer(viewer)
        data['name'] = name
        return cls(data, viewer.progress)

    @classmethod
    def from_file(cls, file_name, progress=ProgressIndicator()):
        """ Create new DBStoredProvider from JSON file
        """
        data = json.load(file_name)
        return cls(data, progress)

    def save(self, file_name):
        """ Save selection to JSON file
        """
        f = open(file_name, 'w')
        json.dump(self.data, f, sort_keys=True, indent=4)
        f.close()

    def data_dict(self):
        """ Return a dictionary with all information to serialize the object
        """
        self.data['name'] = self.name
        return self.data

    def blocks(self):
        """ Return a list of selected Block objects
        """
        if self.block_cache is None:
            self.block_cache = [NeoDataProvider.get_block(b[1], b[0])
                           for b in self.data['blocks']]
        return self.block_cache

    def segments(self):
        """ Return a list of selected Segment objects
        """
        blocks = self.blocks()
        segments = []
        for s in self.data['segments']:
            segments.append(blocks[s[1]].segments[s[0]])
        return segments

    def recording_channel_groups(self):
        """ Return a list of selected
        """
        blocks = self.blocks()
        rcgs = []
        for rcg in self.data['channel_groups']:
            rcgs.append(blocks[rcg[1]].recordingchannelgroups[rcg[0]])
        return rcgs

    def recording_channels(self):
        """ Return a list of selected recording channel indices
        """
        return self.data['channels']

    def units(self):
        """ Return a list of selected Unit objects
        """
        rcgs = self.recording_channel_groups()
        units = []
        for u in self.data['units']:
            units.append(rcgs[u[1]].units[u[0]])
        return units

# Enable automatic creation of NeoStoredProvider objects
DataProvider._factories['Neo'] = NeoStoredProvider