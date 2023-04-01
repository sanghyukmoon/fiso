import numpy as np
import xarray as xr
import time
from fiso import boundary


def filter_var(var, iso_dict=None, cells=None, fill_value=np.nan):
    """Set var = 0 outside the region defined by iso_dict

    Parameters
    ----------
    var : xarray.DataArray
        Variable to be filtered (rho, phi, etc.)
    iso_dict : dict, optional
        FISO object dictionary
    cells : list or array-like, optional
        Flattend indices

    Return
    ------
    out : Filtered DataArray
    """
    if iso_dict is None and cells is None:
        return var
    elif iso_dict is not None and cells is not None:
        raise ValueError("Either give iso_dict or cells, but not both")
    elif cells is None:
        cells = []
        for value in iso_dict.values():
            cells += list(value)
    var_flat = var.data.flatten()
    out = np.full(len(var_flat), fill_value)
    out[cells] = var_flat[cells]
    out = out.reshape(var.shape)
    out = xr.DataArray(data=out, coords=var.coords, dims=var.dims)
    return out


def get_coords_minimum(arr):
    """returns coordinates at the minimum of arr

    Args:
        arr : xarray.DataArray instance
    Returns:
        x0, y0, z0
    """
    center = arr.argmin(...)
    x0, y0, z0 = [arr.isel(center).coords[dim].data[()]
                  for dim in ['x', 'y', 'z']]
    return x0, y0, z0


def timer(message="", verbose=True):
    """Measure wall clock time

    Arguments
    ---------
    message: string, optional
        Short description of the job to measure time
    verbose: boolean, optional
        if True, print out measured time

    Return
    ------
    Etot: float
        Total energy-like contained in the isocontour level
    """
    if message=="":
        time.prevtime = time.time()
        return
    thistime = time.time()
    dt = thistime - time.prevtime
    time.prevtime = thistime
    if len(message) > 0 and verbose:
        print("{}: {} seconds elapsed".format(message, dt))
    return dt
