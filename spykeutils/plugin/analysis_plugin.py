import hashlib
import json
import os
import tables
import time

import spykeutils.plugin.gui_data as gui_data
#import guidata.dataset.datatypes as guidata

class HashEntry(tables.IsDescription):
    hash = tables.StringCol(32)
    filename = tables.StringCol(992) # 1024-32 -> long filenames are possible

class AnalysisPlugin(gui_data.DataSet):
    """ Base class for Analysis plugins. Inherit this class to create a
    plugin.

    The two most important methods are :func:`get_name` and :func:`start`.
    Both should be overridden by every plugin. The class also has
    functionality for GUI configuration and saving/restoring analysis
    results.

    The GUI configuration uses :mod:`guidata`. Because `AnalysisPlugin`
    inherits from `DataSet`,
    configuration options can easily be added directly to the class
    definition. For example, the following code creates an analysis that
    has two configuration options which are used in the start() method
    to print to the console::

        class ExampleAnalysis():
            some_time = di.FloatItem('Some time', default=2.0, unit='ms')
            print_more = di.BoolItem('Print additional info', default=True)

            def start(self):
                print 'The selected time is', some_time, 'milliseconds.'
                if print_more:
                    print 'This is important additional information!'


    The class attribute ``data_dir`` contains a base directory for saving
    and loading data. It is set by Spyke Viewer to the directory specified
    in the settings. When using an AnalysisPlugin without Spyke Viewer,
    the default value is an empty string (so the current directory will
    be used) and the attribute can be set to an arbitrary directory.
    """

    data_dir = ''

    def __init__(self):
        super(AnalysisPlugin, self).__init__()
        self.__current = None
        self.__selections = None

    def get_name(self):
        """ Return the name of an analysis. Override to specify analysis
        name.

        :returns: The name of the plugin.
        :rtype: str
        """
        return 'Prototype Analysis'

    def start(self, current, selections):
        """ Entry point for processing. Override with analysis code.

        :param current: This data provider is used if the analysis
            should be performed on the data currently selected in the GUI.
        :type current:
            :class:`spykeviewer.plugin_framework.data_provider.DataProvider`
        :param list selections: This parameter contains all saved
            selections. It is used if an analysis needs multiple data sets.
        """
        pass

    def configure(self):
        """ Configure the analysis. Override if a different or additional
        configuration apart from guidata is needed.
        """
        if self._items:
            self.edit()

    def get_parameters(self):
        """ Return a dictionary of the configuration that can
        be read with :func:`deserialize_parameters`. Override both if
        non-guidata attributes need to be serialized or if some guidata
        parameters should not be serialized (e.g. they only affect the
        visual presentation).

        :returns: A dictionary of all configuration parameters.
        :rtype: dict
        """
        if not hasattr(self, '_items'):
            return {}

        ret = {}
        for i in self._items:
            v = i.get_value(self)
            if isinstance(v, str):
                ret[i._name] = unicode(v)
            else:
                ret[i._name] = v
        return ret

    def set_parameters(self, parameters):
        """ Load configuration from a dictionary that has been
        created by :func:`serialize_parameters`. Override both if
        non-guidata attributes need to be serialized or if some guidata
        parameterss hould not be serialized (e.g. they only affect the
        visual presentation).

        :param dict parameters: A dictionary of all configuration
            parameters.
        """
        for n,v in parameters.iteritems():
            setattr(self, '_' + n, v)


    def _get_hash(self, selections, params, use_guiparams):
        """ Return hash and the three strings used for it
        (guidata,selections,params)
        """
        if use_guiparams:
            guidata_string = repr(sorted(self.get_parameters().items()))
        else:
            guidata_string = ''
        selection_string = json.dumps([s.data_dict() for s in selections])

        if params:
            param_string = repr(sorted(params.items()))
        else:
            param_string = ''

        md5 = hashlib.md5()
        hash_string = guidata_string + selection_string + param_string
        md5.update(hash_string)

        return md5.hexdigest(), guidata_string, selection_string, \
               param_string

    def save(self, name, selections, params=None, save_guiparams=True):
        """ Return a HDF5 file object with parameters already stored.
        Save analysis results to this file.

        :param str name: The name of the results to save. A folder with
            this name will be used (and created if necessary) to store
            the analysis result files.
        :param sequence selections: A list of :class:`DataProvider` objects
            that are relevant for the analysis results.
        :param dict params: A dictionary, indexed by strings (which should
            be valid as python identifiers), with parameters apart from GUI
            configuration used to obtain the results. All keys have to be
            integers, floats, strings or lists of these types.
        :param bool save_guiparams: Determines if the guidata parameters of
            the class should be saved in the file.
        :returns: An open PyTables file object ready to be used to store
            data. Afterwards, the file has to be closed by calling the
            :func:`tables.File.close` method.
        :rtype: :class:`tables.File`
        """
        if not selections:
            selections = []

        if not os.path.exists(os.path.join(self.data_dir, name)):
            os.makedirs(os.path.join(self.data_dir, name))

        if params is None:
            params = {}

        # Use unicode parameters
        for n, v in params:
            if isinstance(v, str):
                params[n] = unicode(v)

        # Create parameter hash
        hash_, guidata_string, selection_string, param_string =\
        self._get_hash(selections, params, save_guiparams)

        # File name is current time stamp
        time_stamp = time.strftime("%Y%m%d-%H%M%S")
        file_name_base = os.path.join(self.data_dir, name, time_stamp)
        file_name = file_name_base

        # Make sure not to overwrite another file
        i = 2
        while os.path.exists(file_name):
            file_name = file_name_base + '_%d' % i
            i += 1
        file_name += '.h5'

        self._add_hash_lookup_entry(name, hash_, file_name)

        h5 = tables.openFile(file_name, 'w')

        # Save guidata parameters
        paramgroup = h5.createGroup('/', 'guiparams')
        guiparams = self.get_parameters()
        for p,v in guiparams.iteritems():
            t = type(v)
            if t == int or t == float:
                h5.setNodeAttr(paramgroup, p, v)
            else:
                h5.setNodeAttr(paramgroup, p, json.dumps(v))

        # Save selections the provided by plugin
        h5.setNodeAttr('/', 'selections', selection_string)

        # Save additional parameters provided by plugin
        paramgroup = h5.createGroup('/', 'userparams')
        for p,v in params.iteritems():
            t = type(v)
            if t == int or t == float:
                h5.setNodeAttr(paramgroup, p, v)
            else:
                h5.setNodeAttr(paramgroup, p, json.dumps(v))

        # Save hash and current time
        h5.setNodeAttr('/', '_hash', hash_)
        h5.setNodeAttr('/', 'time', time.time())

        return h5

    def load(self, name, selections, params=None, consider_guiparams=True):
        """ Return the most recent HDF5 file for a certain parameter
        configuration. If no such file exists, return None. This
        function works with the files created by :func:`save`.

        :param str name: The name of the results to load.
        :param sequence selections: A list of :class:`DataProvider` objects
            that are relevant for the analysis results.
        :param dict params: A dictionary, indexed by strings (which should
            be valid as python identifiers), with parameters apart from GUI
            configuration used to obtain the results. All keys have to be
            integers, floats, strings or lists of these types.
        :param bool consider_guiparams: Determines if the guidata parameters
            of the class should be considered if they exist in the HDF5
            file. This should be set to False if :func:`save` is used with
            ``save_guiparams`` set to ``False``.
        :returns: An open PyTables file object ready to be used to read
            data. Afterwards, the file has to be closed by calling the
            :func:`tables.File.close` method. If no appropriate file
            exists, None is returned.
        :rtype: :class:`tables.File`
        """
        if not selections:
            selections = []

        if not os.path.exists(os.path.join(self.data_dir, name)):
            return None

        if params is None:
            params = {}

        # Use unicode parameters
        for n, v in params:
            if isinstance(v, str):
                params[n] = unicode(v)

        hash_, guidata_string, selection_string, param_string =\
            self._get_hash(selections, params, consider_guiparams)

        # Loop through files and find the most recent match
        file_names = self._get_hash_file_names(name, hash_)
        newest = 0.0
        best = None
        for fn in file_names:
            with tables.openFile(fn, 'r') as h5:
                file_hash = h5.getNodeAttr('/', '_hash')

                if hash_ != file_hash:
                    continue

                # Hash is correct, check guidata parameters
                gui_params = {}
                for pname in h5.root.guiparams._v_attrs._f_list('user'):
                    v = h5.getNodeAttr('/guiparams', pname)
                    if isinstance(v, str):
                        gui_params[pname] = json.loads(v)
                    else:
                        gui_params[pname] = v

                if gui_params:
                    gui_param_string = repr(sorted(gui_params.items()))
                else:
                    gui_param_string = ''

                if gui_param_string != guidata_string:
                    continue

                # Check selections
                file_selections = h5.getNodeAttr('/', 'selections')
                if file_selections != selection_string:
                    continue

                # Check custom parameters
                file_params = {}
                for pname in h5.root.userparams._v_attrs._f_list('user'):
                    v = h5.getNodeAttr('/userparams', pname)
                    if isinstance(v, str):
                        file_params[pname] = json.loads(v)
                    else:
                        file_params[pname] = v

                if file_params:
                    file_param_string = repr(sorted(file_params.items()))
                else:
                    file_param_string = ''

                if file_param_string != param_string:
                    continue

                # Make sure the most recent file is used
                analysis_time = h5.getNodeAttr('/', 'time')
                if analysis_time < newest:
                    continue

                best = fn
                newest = analysis_time

        if best:
            return tables.openFile(best, 'r')
        return None

    @classmethod
    def _create_hash_lookup_file(cls, name):
        """ (Re)creates a hash lookup file for a results directory. This
        file contains all file hashes in the directory so that the
        correct file for a given parameter set can be found quickly.

        :param str name: The name of the results.
        """
        name = os.path.join(cls.data_dir, name)
        hashfile_name = os.path.join(name, 'hash.h5')
        hash_file = tables.openFile(hashfile_name, mode='w')
        table = hash_file.createTable('/', 'lookup_table', HashEntry,
            title='Hash lookup')

        # Loop through files and write hashes
        file_names = [os.path.join(name,f) for f in os.listdir(name)]
        entry = table.row
        for fn in file_names:
            if not fn.endswith('.h5') or fn == 'hash.h5':
                continue

            try:
                with tables.openFile(fn, 'r') as h5:
                    file_hash = h5.getNodeAttr('/', '_hash')
                    entry['hash'] = file_hash
                    entry['filename'] = fn
                    entry.append()
            except Exception:
                pass # Not a valid data file, no problem

        hash_file.close()

    @classmethod
    def _add_hash_lookup_entry(cls, name, hash_, file_name):
        """ Add a new entry to the hash lookup file.

        :param str name: The name of the results.
        :param str hash_: The hash of the parameters.
        :param str file_name: The file name of the results.
        """
        hashfile_name = os.path.join(cls.data_dir, name, 'hash.h5')
        if not os.path.exists(hashfile_name):
            cls._create_hash_lookup_file(name)

        hash_file = tables.openFile(hashfile_name, mode='r+')
        table = hash_file.root.lookup_table

        # Add entry
        entry = table.row
        entry['hash'] = hash_
        entry['filename'] = file_name
        entry.append()

        hash_file.close()

    @classmethod
    def _get_hash_file_names(cls, name, hash_, _recurse=False):
        """ Return a list of file names for a parameter hash. If no hash
        lookup file exists, it will be created. If it can not be
        created, a list HDF5 files in the directory will be returned.

        :param str name: The name of the results.
        :param str hash_: The hash of the parameters.
        :param bool _recurse: Internal guard against infinite recursion.
        """
        dataname = name
        name = os.path.join(cls.data_dir, name)
        hashfile_name = os.path.join(name, 'hash.h5')
        if not os.path.exists(hashfile_name):
            try:
                cls._create_hash_lookup_file(name)
            except Exception:
                return [os.path.join(name,f) for f in os.listdir(name)
                        if f.endswith('.h5') and not f == 'hash.h5']

        hash_file = tables.openFile(hashfile_name, mode='r')
        table = hash_file.root.lookup_table

        files = [row['filename'] for row in
                 table.where('hash == "%s"' % hash_)]

        ret = []
        for f in files:
            if os.path.exists(f):
                ret.append(f)
            elif not _recurse:
                hash_file.close()
                try:
                    cls._create_hash_lookup_file(name)
                except Exception:
                    return [os.path.join(name,f) for f in os.listdir(name)
                            if f.endswith('.h5') and not f == 'hash.h5']
                return cls._get_hash_file_names(dataname, hash_, True)

        hash_file.close()
        return ret