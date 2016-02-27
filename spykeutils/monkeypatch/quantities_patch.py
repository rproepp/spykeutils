from __future__ import absolute_import
import quantities as pq


# At least up to quantities 0.10.1 the additional arguments to the min and max
# function did not get passed along.
# A fix already exists:
# <https://github.com/dopplershift/python-quantities/commit/30e8812ac15f551c65311d808c2a004f53913a25>
# Also a pull request exists which has not been merged at the time of writing
# 01/18/2013:
# <https://github.com/python-quantities/python-quantities/pull/41>

def _Quanitity_max(self, axis=None, out=None):
    return pq.Quantity(
        self.magnitude.max(axis, out),
        self.dimensionality,
        copy=False
    )
pq.Quantity.max = _Quanitity_max


def _Quanitity_min(self, axis=None, out=None):
    return pq.Quantity(
        self.magnitude.min(axis, out),
        self.dimensionality,
        copy=False
    )
pq.Quantity.min = _Quanitity_min


# Python quantities does not use have additional parameters for astype()
# which became a problem in linspace in numpy 1.11. This is a dirty, dirty
# hack to allow the Quantity astype function to accept any arguments and work
# with numpy >= 1.11. A bug has been filed at
# <https://github.com/python-quantities/python-quantities/issues/105>
_original_astype = pq.Quantity.astype
def _Quantity_astype(self, dtype=None, *args, **kwargs):
    return _original_astype(self, dtype)
pq.Quantity.astype = _Quantity_astype
