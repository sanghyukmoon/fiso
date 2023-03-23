import numpy as np
from scipy.ndimage import minimum_filter
import time
from .edge import precompute_neighbor

# printing extra diagnostic messages
verbose = True
# how to determine neighbors of boundary cells
boundary_flag = 'periodic'
# whether diagonal cells are neighbors
time.prevtime = time.time()


def timer(string=''):
    thistime = time.time()
    dt = thistime - time.prevtime
    time.prevtime = thistime
    if len(string) > 0:
        if verbose: print(string,str(dt) + " seconds elapsed")
    return dt


def setup(data):
    timer()
    # Prepare data
    data_flat = data.flatten()
    order = data_flat.argsort() # sort phi in ascending order.
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
    return idx_minima, order, len(order), pcn


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


def find_minima_no_bc(arr):
    '''
    Find minima using sn, don't allow any boundary cells to be minima
    Then add the boundaries
    '''
    # initially doesnt allow any boundary cells to be minima
    local_min = (arr == minimum_filter(arr, size=3, mode='constant',
                                       cval=-np.inf)).flatten()
    return local_min


def find_minima_boundary_only(data_flat,indices,bcn):
    '''
    data_flat: 1-d array data
    indices: 1-d array of boundary flattened indices
    bcn: boundary neighbors: num_indices x num_neighbors 2-d array of
    flattened indices
    output: indices that are local minima.
    '''
    indices = np.array(indices)
    return indices[data_flat[indices] <= np.min(data_flat[bcn],axis=1)]
