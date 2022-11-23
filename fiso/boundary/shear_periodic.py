# Assumes axis 0 is to be sheared in the axis 1 direction
# Assumes axis 0 and 1 are shear-periodic, axis 2 is

import numpy as np
import fiso.fiso as fiso

boundary_mode = ['wrap','wrap','clip']
corner_bool = True

def boundary_neighbors(shape,cell_shear):
    nps = np.prod(shape)
    dim = len(shape)
    #save on memory when applicable
    if nps < 2**31:
        dtype = np.int32
    else:
        dtype = np.int64
    # get the boundary indices of x = 0 and x = shape[0][-1]
    bound_axis = 0
    shear_axis = 1
    face0,face1 = fiso.gbi_axis(shape,dtype,bound_axis)
    # calculate the coords of those indices
    bc0 = np.array(np.unravel_index(face0,shape),dtype=dtype)
    bc1 = np.array(np.unravel_index(face1,shape),dtype=dtype)
    itp = fiso.calc_itp(dim,corner_bool,dtype)
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
    bcn0 = list(np.ravel_multi_index(nc0,shape,mode=boundary_mode))
    bcn1 = list(np.ravel_multi_index(nc1,shape,mode=boundary_mode))
    face = np.array(face0 + face1)
    bcn = np.array(bcn0 + bcn1)

    # all faces, boundary neighbors
    bi,bpcn = fiso.boundary_i_bcn(shape,dtype,itp,corner_bool,boundary_mode)
    bi,bu = np.unique(bi,return_index=True)
    bpcn = bpcn[bu]
    # correct shear face and bcn
    bi_as = np.argsort(bi)
    # bi_as[n] is the original position of nth element
    bi_as_face = np.searchsorted(bi[bi_as],face)
    # bi_as_face[i] is the n position of face[i]
    bpcn[bi_as[bi_as_face]] = bcn
    # is original positions of face
    # return all faces boundary neighbors
    return bi,bpcn

def make_pcn_mem(shape,cell_shear):
    '''
    Memory efficient version of make_pcn, calculates bulk neighbors on the fly
    but precomputes boundary neighbors.

    Arguments:
    shape: NumPy array shape
    cell_shear: integer

    Also depends on:
    fiso.boundary.shear_periodic.boundary_mode
    fiso.boundary.shear_periodic.cornerbool

    Returns:
    bi: boundary indices 1-D flattened
    bpcn: boundary pre-computed neighbors shape (len(bi),len(neighbors per cell))
    pcn: pcn[i] returns the neighbor indices of cell i.

    '''
    
    bi,bpcn = boundary_neighbors(shape,cell_shear)
    displacements = fiso.compute_displacement(shape,corner_bool)
    class pcnDict(dict):
        def __getitem__(self,index):
            return self.get(index,index+displacements)
    pcn = pcnDict(zip(bi,bpcn))
    return bi,bpcn,pcn

def make_pcn(shape,cell_shear):
    '''
    Memory inefficient version of make_pcn_mem,
    precomputes all neighbors for all cells and stores.

    Arguments:
    shape: NumPy array shape
    cell_shear: integer

    Also depends on:
    fiso.boundary.shear_periodic.boundary_mode
    fiso.boundary.shear_periodic.cornerbool

    Returns:
    bi: boundary indices 1-D flattened
    bpcn: boundary pre-computed neighbors shape (len(bi),len(neighbors per cell))
    pcn: pcn[i] returns the neighbor indices of cell i.

    '''
    
    bi,bpcn = boundary_neighbors(shape,cell_shear)
    pcn = fiso.precompute_neighbor(shape,corner=corner_bool,mode=boundary_mode)
    pcn[bi] = bpcn
    return bi,bpcn,pcn
