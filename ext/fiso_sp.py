# Shear Periodic Boundary Conditions
# X, Y periodic.
# Z clip
from fiso import fiso
import numpy as n

boundary_mode = ['wrap','wrap','clip']
corner_bool = True
cell_shear = 0
# add shear to Y when X = X_min, X_max
def setup(data,cut):
    dshape = data.shape
    dlist = data.reshape(-1) #COPY
    order = dlist.argsort() #an array of real index locations #COPY
    fiso.timer('sort')
    cutoff = len(order) #number of cells to process
    #optional cutoff
    if type(cut) is float:
        cutoff = n.searchsorted(dlist[order],cut)

    # precompute neighbor indices
    bi, bpcn, pcn = shear_pcn(dshape,cell_shear)
    fiso.timer('precompute neighbor indices')

    # find minima without bc
    # find minima with bc
    # combine
    mfw0 = n.where(fiso.find_minima_no_bc(data).reshape(-1))[0]
    mfw1 = fiso.find_minima_boundary_only(dlist,bi,bpcn)
    mfw = n.unique(n.sort(n.append(mfw0,mfw1)))
    return mfw,order,cutoff,pcn

def shear_bcn(shape,cell_shear):
    nps = n.prod(shape)
    #save on memory when applicable
    if nps < 2**31:
        dtype = n.int32
    else:
        dtype = n.int64
    # get the boundary indices of x = 0 and x = shape[0][-1]
    bound_axis = 0
    shear_axis = 1
    face0,face1 = fiso.gbi_axis(shape,dtype,bound_axis)
    # calculate the coords of those indices
    bc0 = n.array(n.unravel_index(face0,shape),dtype=dtype)
    bc1 = n.array(n.unravel_index(face1,shape),dtype=dtype)
    itp = fiso.calc_itp(3,corner_bool,dtype)
    titp = n.transpose(itp)[:,None,:]
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
    bcn0 = list(n.ravel_multi_index(nc0,shape,mode=boundary_mode))
    bcn1 = list(n.ravel_multi_index(nc1,shape,mode=boundary_mode))
    face = n.array(face0 + face1)
    bcn = n.array(bcn0 + bcn1)

    # all faces, boundary neighbors
    bi,bpcn = fiso.boundary_i_bcn(shape,dtype,itp,corner_bool,boundary_mode)
    bi,bu = n.unique(bi,return_index=True)
    bpcn = bpcn[bu]
    # correct shear face and bcn
    bi_as = n.argsort(bi)
    # bi_as[n] is the original position of nth element
    bi_as_face = n.searchsorted(bi[bi_as],face)
    # bi_as_face[i] is the n position of face[i]
    bpcn[bi_as[bi_as_face]] = bcn
    # is original positions of face
    # return all faces boundary neighbors
    return bi,bpcn

def shear_pcn(shape,cell_shear):
    bi,bpcn = shear_bcn(shape,cell_shear)
    pcn = fiso.precompute_neighbor(shape,corner=corner_bool,mode=boundary_mode)
    pcn[bi] = bpcn
    return bi,bpcn,pcn

fiso.setup = setup
find = fiso.find
