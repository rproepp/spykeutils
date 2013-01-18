from __future__ import absolute_import
import scipy as sp
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


# At least up to quantities 0.10.1 the scipy element-wise minimum and maximum
# function did not work.
# This has been reported upstream as issue #53:
# <https://github.com/python-quantities/python-quantities/issues/53>

def _fix_binary_scipy_function_with_out_param(f):
    def _fixed(x1, x2, out=None):
        if isinstance(x1, pq.Quantity) or isinstance(x2, pq.Quantity):
            x1 = x1 * pq.dimensionless
            x2 = x2 * pq.dimensionless
            if out is None:
                x2 = x2.rescale(x1.units)
                return f(x1.magnitude, x2.magnitude) * x1.units
            else:
                x1 = x1.rescale(out.units)
                x2 = x2.rescale(out.units)
                f(x1.magnitude, x2.magnitude, out.magnitude)
                return out
        else:
            return f(x1, x2, out)
    return _fixed
sp.minimum = _fix_binary_scipy_function_with_out_param(sp.minimum)
sp.maximum = _fix_binary_scipy_function_with_out_param(sp.maximum)


# At least up to quantities 0.10.1 the scipy meshgrid  and concatenate
# functions did lose units.
# This has been reported upstream as issue #47:
# <https://github.com/python-quantities/python-quantities/issues/47>
def _fix_scipy_meshgrid(f):
    def _fixed(x, y):
        rx, ry = f(x, y)
        if isinstance(x, pq.Quantity):
            rx = rx * x.units
        if isinstance(y, pq.Quantity):
            ry = ry * y.units
        return rx, ry
    return _fixed
sp.meshgrid = _fix_scipy_meshgrid(sp.meshgrid)


def _fix_scipy_concatenate(f):
    def _fixed(arrays, axis=0):
        is_quantity = len(arrays) > 0 and isinstance(arrays[0], pq.Quantity)
        if is_quantity:
            arrays = [(a * pq.dimensionless).rescale(arrays[0].units)
                      for a in arrays]
        else:
            for a in arrays:
                if (isinstance(a, pq.Quantity) and
                        a.units.simplified != pq.dimensionless):
                    raise ValueError(
                        'Cannot concatenate arrays of different units')
        concatenated = f(arrays, axis=axis)
        if is_quantity:
            concatenated = concatenated * arrays[0].units
        return concatenated
    return _fixed
sp.concatenate = _fix_scipy_concatenate(sp.concatenate)
