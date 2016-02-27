import scipy as sp
import quantities as pq


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
minimum = _fix_binary_scipy_function_with_out_param(sp.minimum)
maximum = _fix_binary_scipy_function_with_out_param(sp.maximum)


# At least up to quantities 0.10.1 the scipy meshgrid  and concatenate
# functions did lose units.
# This has been reported upstream as issue #47:
# <https://github.com/python-quantities/python-quantities/issues/47>
# Fixed with scipy 0.17
def _fix_scipy_meshgrid(f):
    def _fixed(x, y):
        rx, ry = f(x, y)
        if isinstance(x, pq.Quantity) and not isinstance(rx, pq.Quantity):
            rx = rx * x.units
        if isinstance(y, pq.Quantity) and not isinstance(ry, pq.Quantity):
            ry = ry * y.units
        return rx, ry
    return _fixed
if sp.__version__ < '0.17':
    meshgrid = _fix_scipy_meshgrid(sp.meshgrid)
else:
    meshgrid = sp.meshgrid


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
concatenate = _fix_scipy_concatenate(sp.concatenate)


# At least up to quantities 0.10.1 the scipy inner and diag functions did not
# respect units.
def _fix_binary_scipy_function(f):
    def _fixed(x1, x2):
        if isinstance(x1, pq.Quantity) or isinstance(x2, pq.Quantity):
            x1 = x1 * pq.dimensionless
            x2 = x2 * pq.dimensionless
            x2 = x2.rescale(x1.units)
            return f(x1.magnitude, x2.magnitude) * x1.units
        else:
            return f(x1, x2)
    return _fixed
inner = _fix_binary_scipy_function(sp.inner)


# diag loses units
# Fixed with scipy 0.17
def diag(v, k=0):
    if isinstance(v, pq.Quantity):
        r = sp.diag(v, k)
        return r if isinstance(r, pq.Quantity) else r * v.units
    else:
        return sp.diag(v, k)


# linspace loses unit for only one bin, see bug report
# <https://github.com/python-quantities/python-quantities/issues/55>
# Fixed with scipy 0.17
def linspace(start, stop, num=50, endpoint=True, retstep=False):
    if int(num) == 1 and isinstance(start, pq.Quantity):
        r = sp.linspace(start, stop, num, endpoint, retstep)
        return r if isinstance(r, pq.Quantity) else r * start.units
    else:
        return sp.linspace(start, stop, num, endpoint, retstep)
