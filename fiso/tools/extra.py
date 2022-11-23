# Collection of tools that may or may not be useful

import scipy.ndimage as sn
import numpy as np
import itertools
from ..fiso import *

# Deprecated minima modes
def find_minima_clip(arr):
    nhbd = sn.generate_binary_structure(len(arr.shape),3) #neighborhood
    local_min = (sn.filters.minimum_filter(arr, footprint=nhbd, mode='reflect')==arr)
    return local_min

def find_minima_nocorner(arr):
    nhbd = sn.generate_binary_structure(len(arr.shape),1) #neighborhood
    local_min = (sn.filters.minimum_filter(arr, footprint=nhbd, mode='reflect')==arr)
    return local_min

def find_minima_wrap(arr):
    nhbd = sn.generate_binary_structure(len(arr.shape),3) #neighborhood
    local_min = (sn.filters.minimum_filter(arr, footprint=nhbd, mode='wrap')==arr)
    return local_min

# Potentially very useful in memory limited situation but incompatible with find_minima_pcn

def live_compute_neighbor(shape,corner=True,mode='clip'):
    '''
    classical precomputed neighbors list pcn takes up a large amount of memory
    : num_neighbors x data size
    To save on memory space without sacrificing much speed,
    replace in <function fiso.setup>:
     pcn = precompute_neighbor(dshape,corner=corner_bool,mode=boundary_mode)
     pcn = live_compute_neighbor(dshape,corner=corner_bool,mode=boundary_mode)
    '''
    nps = np.prod(shape)
    #save on memory when applicable
    if nps < 2**31:
        dtype = np.int32
    else:
        dtype = np.int64
    dim = len(shape)
    itp = calc_itp(dim,corner,dtype)
    ishape = shape[::-1]
    factor = np.cumprod(np.append([1],shape[::-1]))[:-1][::-1]
    factor = factor.astype(dtype)
    displacements = (itp*factor).sum(axis=1)
    boundary_indices, bpcn = boundary_i_bcn(shape,dtype,itp,corner,mode)
    class pcnDict(dict):
        def __getitem__(self,index):
            return self.get(index,index+displacements)
    pcn = pcnDict(zip(boundary_indices,bpcn))
    return pcn

def blcn(shape,corner=True,mode='clip'):
    '''
    classical precomputed neighbors list pcn takes up a large amount of memory
    : num_neighbors x data size
    To save on memory space without sacrificing much speed,
    replace in <function fiso.setup>:
     pcn = precompute_neighbor(dshape,corner=corner_bool,mode=boundary_mode)
     pcn = live_compute_neighbor(dshape,corner=corner_bool,mode=boundary_mode)
    '''
    nps = np.prod(shape)
    #save on memory when applicable
    if nps < 2**31:
        dtype = np.int32
    else:
        dtype = np.int64
    dim = len(shape)
    itp = calc_itp(dim,corner,dtype)
    ishape = shape[::-1]
    factor = np.cumprod(np.append([1],shape[::-1]))[:-1][::-1]
    factor = factor.astype(dtype)
    displacements = (itp*factor).sum(axis=1)
    boundary_indices, bpcn = boundary_i_bcn(shape,dtype,itp,corner,mode)
    class pcnDict(dict):
        def __getitem__(self,index):
            return self.get(index,index+displacements)
    pcn = pcnDict(zip(boundary_indices,bpcn))
    return pcn




# Deprecated version of shear periodic neighbors

def shear_periodic_pcn(shape,dtype,cell_shear,boundary_mode,bound_axis=0,shear_axis=1):
    '''
    For shear periodic boundary conditions, compute the indices of the faces
    and compute the indices of their neighbours. 
    '''
    # get the boundary indices of x = 0 and x = shape[0][-1]
    face0,face1 = gbi_axis(shape,dtype,bound_axis)
    # calculate the coords of those indices
    bc0 = np.array(np.unravel_index(face0,shape),dtype=dtype)
    bc1 = np.array(np.unravel_index(face1,shape),dtype=dtype)
    # calculate itp, itertools product [-1,0,1]
    itp = calc_itp(3,True,dtype)
    titp = np.transpose(itp)[:,None,:]
    # get all neighboring coordinates
    nc0 = bc0[:,:,None] + titp
    nc1 = bc1[:,:,None] + titp
    # for coords that are out of range in x, add the appropriate shear in y
    # x end y = 0 belongs to x start y = offset
    nc0[shear_axis][
        nc0[bound_axis] < 0
    ] += shape[shear_axis] - cell_shear
    nc1[shear_axis][
        nc1[bound_axis] == shape[bound_axis]
    ] += cell_shear
    # note that adding extra shape[shear_axis] to y is okay
    # because of wrap boundary mode.
    bcn0 = np.ravel_multi_index(nc0,shape,mode=boundary_mode)
    bcn1 = np.ravel_multi_index(nc1,shape,mode=boundary_mode)
    return face0,bcn0,face1,bcn1
