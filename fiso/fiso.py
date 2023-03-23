import numpy as np
from scipy.ndimage import minimum_filter
from .edge import precompute_neighbor
from .tools import timer
import itertools

# how to determine neighbors of boundary cells
boundary_flag = 'periodic'

#TODO(SMOON) Rearrangment of modules is warranted.

def setup(data, verbose=True):
    # TODO(SMOON) use more meaningful function name
    timer()
    # Prepare data
    data_flat = data.flatten()
    cells_ordered = data_flat.argsort() # sort phi in ascending order.
    timer('sort')

    # Precompute neighbor indices
    pcn = precompute_neighbor(data.shape, corner=True, boundary_flag=boundary_flag)
    timer('precompute neighbor indices')

    # Find local minima
    # minima_flat is flattened boolian mask
    minima_flat = find_minima_global(data, boundary_flag).flatten()
    # 1D flattened index array of potential minima
    idx_minima = np.where(minima_flat)[0]
    if verbose:
        print("Found {} minima".format(len(idx_minima)))
    return idx_minima, cells_ordered, len(cells_ordered), pcn


def find_minima_global(arr, boundary_flag='periodic'):
    """Find local minima of the input array

    Parameters
    ----------
    arr : array_like
        Input array
    boundary_flag: str, optional
        boundary flag determines how the input array is extended when the
        stencil for finding the local minima overlaps a border.

    Returns
    -------
    local_min : array_like
        Bolean array that selects local minima
    """
    if boundary_flag == 'clip':
        mode = 'reflect'
    elif boundary_flag == 'periodic':
        mode = 'wrap'
    else:
        raise Exception("unknown boundary mode")
    local_min = (arr == minimum_filter(arr, size=3, mode=mode))
    return local_min
