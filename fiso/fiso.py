import numpy as np
from scipy.ndimage import minimum_filter
import time
from itertools import islice
import itertools
from collections import deque

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
    minima_flat = find_minima_global(data, boundary_flag).flatten()
    idx_minima = np.where(minima_flat)[0]
    if verbose:
        print("Found {} minima".format(len(idx_minima)))
    return idx_minima, order, len(order), pcn


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


def boundary_pcn(coords, offsets, shape, corner, boundary_flag='periodic'):
    if boundary_flag=='periodic':
        mode='wrap'
    else:
        mode='clip'
    newcoords = coords[:,:,None] + np.transpose(offsets)[:,None,:]
    output = np.ravel_multi_index(newcoords,shape,mode=mode)
    return output
    # start with coords of shape dim,num_coords
    # for each num_coords, add one of the many offsets
    # offsets has shape num_neighbors,dim
    # np.transpose(offsets)
    # dim, num_neighbors
    # want something of shape dim,num_coords,num_neighbors
    # num_coords, num_neighbors

def gbi_axis(shape,dtype,axis):
    '''
    get boundary indices of axis from shape
    useful when boundary condition is axis specific
    '''
    shape = list(shape)
    dim = len(shape)
    basel = dim*[None] #array slice none to extend array dimension
    idx = range(dim) #index for loops
    #dni is the coords for dimension "i"
    dni = dim*[None]
    for i in idx:
        dni[i] = np.arange(shape[i],dtype=dtype)
    #for boundary dimensions i set indices j != i, setting index i to be
    #0 or end

    ndnis = dim*[None]
    i = axis
    shapei = shape[:] #copy shape
    shapei[i] = 1     #set dimension i to 1 (flat boundary)
    nzs = np.zeros(shapei,dtype=dtype) #initialize boundary to 0
    for j in idx:
        if j == i:
            continue
        #make coord j using the np.arange (dni) with nzs of desired shape
        selj = basel[:]
        selj[j] = slice(None)
        #slicing on index j makes dni[j] vary on index j and copy on other dimensions with desired shape nzs
        ndnis[j] = dni[j][tuple(selj)] + nzs
    ndnis[i] = 0
    face0 = list(np.ravel_multi_index(ndnis,shape).flatten())
    ndnis[i] = shape[i]-1
    face1 = list(np.ravel_multi_index(ndnis,shape).flatten())
    return face0,face1

def gbi(shape, dtype):
    """get boundary indices from shape"""
    shape = list(shape)
    bi = []
    ls = len(shape)
    basel = ls*[None] #array slice none to extend array dimension
    idx = range(ls) #index for loops
    #dni is the coords for dimension "i"
    dni = ls*[None]
    for i in idx:
        dni[i] = np.arange(shape[i],dtype=dtype)

    #for boundary dimensions i set indices j != i, setting index i to be
    #0 or end
    for i in idx:

        ndnis = ls*[None]
        shapei = shape[:] #copy shape
        shapei[i] = 1     #set dimension i to 1 (flat boundary)
        nzs = np.zeros(shapei,dtype=dtype) #initialize boundary to 0
        for j in idx:
            if j == i:
                continue
                #make coord j using the np.arange (dni) with nzs of desired shape
            selj = basel[:]
            selj[j] = slice(None)
            #slicing on index j makes dni[j] vary on index j and copy on other dimensions with desired shape nzs
            ndnis[j] = dni[j][tuple(selj)] + nzs
        ndnis[i] = 0
        bi += list(np.ravel_multi_index(ndnis,shape).flatten())
        ndnis[i] = shape[i]-1
        bi += list(np.ravel_multi_index(ndnis,shape).flatten())
    return bi

def get_offsets(dim, corner=True):
    """Compute 1-D flattened array offsets corresponding to neighbors

    Parameters
    ----------
    dim : int
        dimension of the input data
    corner : Boolean, optional
        If true, the corner cells are counted as neighbors (26 neighbors in total)

    Returns
    -------
    offsets : array-like
        The shape of this array is (N_neighbors, dim). Each rows are integer
        directions that points to the neighbor cells. For example, in 3D, the
        offsets will look like:
        (-1, -1, -1)
        (-1, -1,  0)
        (-1, -1,  1)
        (-1,  0, -1)
        (-1,  0,  0)
        (-1,  0,  1)
        (-1,  1, -1)
        (-1,  1,  0)
        (-1,  1,  1)
        ( 0, -1, -1)
        ...
    """
    offs = [-1,0,1]
    offsets = list(itertools.product(offs,repeat=dim))
    if corner:
        offsets.remove((0,)*dim)
    else:
        offsets = [i for i in offsets if i.count(0) == 2]
    offsets = np.array(offsets, dtype=np.int32))
    return offsets

def precompute_neighbor(shape,corner=True,boundary_flag='periodic'):
    """Precompute neighbor indices

    Parameters
    ----------
    shape : tuple
        shape of the input data
    corner : Boolean, optional
        If true, the corner cells are counted as neighbors (26 neighbors in total)
    boundary_flag: str, optional
        Flag for boundary condition. Affects how to set neighbors of the edge cells.

    Returns
    -------
    pcn : array_like
        The shape of this array is (Ncells, N_neighbors), such that pcn[1][0] is
        the flattened index of the (-1,-1,-1) neighbor of the (k,j,i) = (0,0,1)
        cell, which is (k,j,i) = (-1, -1, 0) = (Nz-1, Ny-1, 0) for periodic BC.
        See docstring of get_offsets for the ordering of neighbor directions. 
    """
    Ncells = np.prod(shape)
    # Save on memory when applicable
    if Ncells < 2**31:
        dtype = np.int32
    else:
        dtype = np.int64
    # Set up offset array
    dim = len(shape)
    offsets = get_offsets(dim, corner, dtype)
    # Set up displacements in index space (treat n-d array as 1-d list)
    # factor = (NxNy, Nx, 1)
    factor = np.cumprod(np.append([1],shape[::-1]))[:-1][::-1]
    factor = factor.astype(dtype)
    displacements = (offsets*factor).sum(axis=1)
    # Displacements is num_neighbors 1-d array
    indices = np.arange(Ncells, dtype=dtype)[:,None]
    # indices is 1-d array, 1 for each cell
    pcn = indices + displacements[None]
    # pcn is 2-d array using :,None to combine
    # Apply boundary correction
    bi,bpcn = boundary_i_bcn(shape, dtype, offsets, corner, boundary_flag)
    # pcn shape num_neighbors x cells
    # bcn shape num_neighbors x cells
    pcn[bi] = bpcn
    return pcn
    
def boundary_i_bcn(shape, dtype, offsets, corner, boundary_flag):
    # returns boundary indices and boundary's neighbor indices.
    boundary_indices = gbi(shape,dtype)
    boundary_coords = np.array(np.unravel_index(boundary_indices,shape),
                              dtype=dtype)
    bpcn = boundary_pcn(boundary_coords, offsets, shape, corner,
                        boundary_flag=boundary_flag).astype(dtype)
    return boundary_indices,bpcn
