import numpy as np
from scipy.ndimage import minimum_filter
import time
from itertools import islice
import itertools
from collections import deque

# printing extra diagnostic messages
verbose = True
# how to determine neighbors of boundary cells
boundary_mode = 'periodic'
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
    #prepare data
    dshape = data.shape
    data_flat = data.flatten()
    order = data_flat.argsort() # sort phi in ascending order.
    timer('sort')
    cutoff = len(order) #number of cells to process

    #precompute neighbor indices
    pcn = precompute_neighbor(dshape,corner=True,boundary_mode=boundary_mode)
    timer('precompute neighbor indices')

    #timer('init short')
    minima_flat = find_minima_global(data, boundary_mode).flatten()
    #indices of the minima in original
    mfw = np.where(minima_flat)[0]
    #mfw is real index
    if verbose: print(len(mfw),'minima')
    return mfw,order,cutoff,pcn


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


def find_minima_global(arr, boundary_mode='periodic'):
    """Find local minima of the input array

    Parameters
    ----------
    arr : array_like
        Input array
    boundary_mode: str, optional
        boundary mode determines how the input array is extended when the
        stencil for finding the local minima overlaps a border.

    Returns
    -------
    local_min : array_like
        Bolean array that selects local minima
    """
    if boundary_mode == 'clip':
        mode0 = 'reflect'
    elif boundary_mode == 'periodic':
        mode0 = 'wrap'
    else:
        raise Exception("unknown boundary mode")
    local_min = (arr == minimum_filter(arr, size=3, mode=mode0))
    return local_min


def boundary_pcn(coords,itp,shape,corner,mode='clip'):
    newcoords = coords[:,:,None] + np.transpose(itp)[:,None,:]
    output = np.ravel_multi_index(newcoords,shape,mode=mode)
    return output
    # start with coords of shape dim,num_coords
    # for each num_coords, add one of the many itp
    # itp has shape num_neighbors,dim
    # np.transpose(itp)
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

def gbi(shape,dtype):
    '''get boundary indices from shape'''
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

def calc_itp(dim,corner,dtype):
    # Compute 1-D flattened array offsets corresponding to neighbors 
    offs = [-1,0,1]
    itp = list(itertools.product(offs,repeat=dim))
    if corner:
        itp.remove((0,)*dim)
    else:
        itp = [i for i in itp if i.count(0) == 2]
    itp = np.array(itp,dtype=dtype)
    return itp

def precompute_neighbor(shape,corner=True,boundary_mode='clip'):
    if boundary_mode=='periodic':
        mode='wrap'
    else:
        mode='clip'
    nps = np.prod(shape)
    #save on memory when applicable
    if nps < 2**31:
        dtype = np.int32
    else:
        dtype = np.int64
    #set up array of cartesian displacements (itp)
    dim = len(shape)
    itp = calc_itp(dim,corner,dtype)
    #set up displacements in index space (treat n-d array as 1-d list)
    ishape = shape[::-1]
    factor = np.cumprod(np.append([1],shape[::-1]))[:-1][::-1]
    factor = factor.astype(dtype)
    displacements = (itp*factor).sum(axis=1)
    #displacements is num_neighbors 1-d array
    indices = np.arange(nps,dtype=dtype)[:,None]
    #indices is 1-d array, 1 for each cell
    pcn = indices + displacements[None]
    #pcn is 2-d array using :,None to combine
    #apply boundary correction mode='clip' set in boundary_pcn
    bi,bpcn = boundary_i_bcn(shape,dtype,itp,corner,mode)
    #pcn shape num_neighbors x cells
    #bcn shape num_neighbors x cells
    pcn[bi] = bpcn
    return pcn
    
def boundary_i_bcn(shape,dtype,itp,corner,mode):
    # returns boundary indices and boundary's neighbor indices.
    boundary_indices = gbi(shape,dtype)
    boundary_coords = np.array(np.unravel_index(boundary_indices,shape),
                              dtype=dtype)
    bpcn = boundary_pcn(boundary_coords,itp,shape,corner,mode=mode).astype(dtype)
    return boundary_indices,bpcn
