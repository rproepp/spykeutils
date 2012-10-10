""" This module gives access to all members of
:mod:`guidata.dataset.dataitems` and :mod:`guidata.dataset.datatypes`.
If :mod:`guidata` cannot be imported, the module offers suitable dummy
objects instead (e.g. for use on a server).
"""
try:
    from guidata.dataset.dataitems import *
    from guidata.dataset.datatypes import *
except ImportError, e:
    import datetime
    import scipy as sp

    # datatypes dummies
    class DataItem(object):
        def __init__(self, *args, **kwargs):
            self._default = None
            if 'default' in kwargs:
                self._default = kwargs['default']

        def get_prop(self, realm, name, default=None):
            pass

        def get_prop_value(self, realm, instance, name, default=None):
            pass

        def set_prop(self, realm, **kwargs):
            return self

        def set_pos(self, col=0, colspan=None):
            return self

        def get_help(self, instance):
            pass

        def get_auto_help(self, instance):
            pass

        def format_string(self, instance, value, fmt, func):
            pass

        def get_string_value(self, instance):
            pass

        def set_name(self, new_name):
            self._name = new_name

        def set_from_string(self, instance, string_value):
            pass

        def set_default(self, instance):
            self.__set__(instance, self._default)

        def accept(self, visitor):
            pass

        def __set__(self, instance, value):
            setattr(instance, "_"+self._name, value)

        def __get__(self, instance, klass):
            if instance is not None:
                return getattr(instance, "_"+self._name, self._default)
            else:
                return self

        def get_value(self, instance):
            return self.__get__(instance, instance.__class__)

        def check_item(self, instance):
            pass

        def check_value(self, instance, value):
            pass

        def from_string(self, instance, string_value):
            pass

        def bind(self, instance):
            return DataItemVariable(self, instance)

        def serialize(self, instance, writer):
            pass

        def deserialize(self, instance, reader):
            pass

    class DataSetMeta(type):
        """ DataSet metaclass.

        Create class attribute `_items`: list of the DataSet class attributes.
        Also make sure that all data items have the correct `_name` attribute.
        """
        def __new__(mcs, name, bases, dct):
            items = []
            for base in bases:
                if getattr(base, "__metaclass__", None) is DataSetMeta:
                    for item in base._items:
                        items.append(item)

            for attrname, value in dct.items():
                if isinstance(value, DataItem):
                    value.set_name(attrname)
                    items.append(value)
            dct["_items"] = items
            return type.__new__(mcs, name, bases, dct)

    class DataSet(object):
        __metaclass__ = DataSetMeta

        def __init__(self, title=None, comment=None, icon=''):
            self.set_defaults()

        def _get_translation(self):
            pass

        def _compute_title_and_comment(self):
            pass

        def get_title(self):
            pass

        def get_comment(self):
            pass

        def get_icon(self):
            pass

        def set_defaults(self):
            for item in self._items:
                item.set_default(self)

        def check(self):
            pass

        def text_edit(self):
            pass

        def edit(self, parent=None, apply=None):
            pass

        def view(self, parent=None):
            pass

        def to_string(self, debug=False, indent=None, align=False):
            pass

        def accept(self, vis):
            pass

        def serialize(self, writer):
            pass

        def deserialize(self, reader):
            pass

        def read_config(self, conf, section, option):
            pass

        def write_config(self, conf, section, option):
            pass

        @classmethod
        def set_global_prop(cls, realm, **kwargs):
            pass

    class ItemProperty(object):
        def __init__(self, callable=None):
            pass

        def __call__(self, instance, item, value):
            pass

        def set(self, instance, item, value):
            pass

    class FormatProp(ItemProperty):
        def __init__(self, fmt, ignore_error=True):
            pass

    class GetAttrProp(ItemProperty):
        pass

    class ValueProp(ItemProperty):
        pass

    class NotProp(ItemProperty):
        pass

    class DataItemVariable(object):
        def __init__(self, item, instance):
            self.item = item
            self.instance = instance

        def get_prop_value(self, realm, name, default=None):
            pass

        def get_prop(self, realm, name, default=None):
            pass

        def get_help(self):
            pass

        def get_auto_help(self):
            pass

        def get_string_value(self):
            pass

        def set_default(self):
            return self.item.set_default(self.instance)

        def get(self):
            return self.item.get_value(self.instance)

        def set(self, value):
            return self.item.__set__(self.instance, value)

        def set_from_string(self, string_value):
            pass

        def check_item(self):
            pass

        def check_value(self, value):
            pass

        def from_string(self, string_value):
            pass

        def label(self):
            pass


    class GroupItem(DataItem):
        pass

    class BeginGroup(DataItem):
        pass

    class EndGroup(DataItem):
        pass

    class TabGroupItem(GroupItem):
        pass

    class BeginTabGroup(BeginGroup):
        def get_group(self):
            pass

    class EndTabGroup(EndGroup):
        pass


    # dataitems dummies
    class NumericTypeItem(DataItem):
        def __init__(self, *args, **kwargs):
            super(NumericTypeItem, self).__init__(*args, **kwargs)

    class FloatItem(NumericTypeItem):
        def __init__(self, *args, **kwargs):
            super(FloatItem, self).__init__(*args, **kwargs)
            if self._default is None:
                self._default = float()

    class IntItem(NumericTypeItem):
        def __init__(self, *args, **kwargs):
            super(IntItem, self).__init__(*args, **kwargs)
            if self._default is None:
                self._default = int()

    class StringItem(DataItem):
        def __init__(self, *args, **kwargs):
            super(StringItem, self).__init__(*args, **kwargs)
            if self._default is None:
                self._default = str()

    class BoolItem(DataItem):
        def __init__(self, *args, **kwargs):
            super(BoolItem, self).__init__(*args, **kwargs)
            if self._default is None:
                self._default = bool()

    class DateItem(DataItem):
        def __init__(self, *args, **kwargs):
            super(DateItem, self).__init__(*args, **kwargs)
            if self._default is None:
                self._default = datetime.date.today()

    class DateTimeItem(DateItem):
        def __init__(self, *args, **kwargs):
            super(DateTimeItem, self).__init__(*args, **kwargs)
            if self._default is None:
                self._default = datetime.datetime.now()

    class ColorItem(StringItem):
        def __init__(self, *args, **kwargs):
            super(ColorItem, self).__init__(*args, **kwargs)
            if self._default is None:
                self._default = str()

    class FileSaveItem(StringItem):
        def __init__(self, *args, **kwargs):
            super(FileSaveItem, self).__init__(*args, **kwargs)

    class FileOpenItem(FileSaveItem):
        def __init__(self, *args, **kwargs):
            super(FileOpenItem, self).__init__(*args, **kwargs)

    class FilesOpenItem(FileSaveItem):
        def __init__(self, *args, **kwargs):
            super(FilesOpenItem, self).__init__(*args, **kwargs)
            if self._default is None:
                self._default = list()

    class DirectoryItem(StringItem):
        def __init__(self, *args, **kwargs):
            super(DirectoryItem, self).__init__(*args, **kwargs)

    class ChoiceItem(DataItem):
        def __init__(self, *args, **kwargs):
            super(ChoiceItem, self).__init__(*args, **kwargs)
            if self._default is None:
                self._default = int()

    class MultipleChoiceItem(ChoiceItem):
        def __init__(self, *args, **kwargs):
            super(MultipleChoiceItem, self).__init__(*args, **kwargs)
            if self._default is None:
                self._default = list()

    class ImageChoiceItem(ChoiceItem):
        def __init__(self, *args, **kwargs):
            super(ImageChoiceItem, self).__init__(*args, **kwargs)

    class FloatArrayItem(DataItem):
        def __init__(self, *args, **kwargs):
            super(FloatArrayItem, self).__init__(*args, **kwargs)
            if self._default is None:
                self._default = sp.array([])

    class ButtonItem(DataItem):
        def __init__(self, *args, **kwargs):
            super(ButtonItem, self).__init__(*args, **kwargs)

    class DictItem(ButtonItem):
        def __init__(self, *args, **kwargs):
            super(DictItem, self).__init__(*args, **kwargs)
            if self._default is None:
                self._default = dict()

    class FontFamilyItem(StringItem):
        def __init__(self, *args, **kwargs):
            super(FontFamilyItem, self).__init__(*args, **kwargs)
            if self._default is None:
                self._default = str()