# Shear Periodic Boundary Conditions
# X, Y periodic.
# Z clip
import fiso
from itertools import islice
import itertools
from collections import deque
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

    #precompute neighbor indices
    pcn = fiso.precompute_neighbor(dshape,corner=corner_bool,mode=boundary_mode)
    fiso.timer('precompute neighbor indices')
    dtype = type(pcn[0,0])

    # correct X endpoints with Y shear
    face0,bcn0,face1,bcn1 = correct_pcn(dshape,dtype,cell_shear)
    pcn[face0] = bcn0
    pcn[face1] = bcn1

    mfw = n.where(fiso.find_minima_pcn(dlist,pcn))[0]
    fiso.timer('minima')

    #core dict and labels setup
    core_dict = {}
    labels = -n.ones(len(order),dtype=int) #indices are real index locations
    #inside loop, labels are accessed by labels[order[i]]
    for mini in mfw:
        core_dict[mini] = deque([mini])
    labels[mfw] = mfw
    active_cores = list(mfw) #real index
    fiso.timer('init minima')
    return core_dict,labels,active_cores,order,cutoff,pcn

def correct_pcn(shape,dtype,cell_shear):
    # get the boundary indices of x = 0 and x = shape[0][-1]
    bound_axis = 0
    shear_axis = 1
    face0,face1 = fiso.gbi_axis(shape,dtype,bound_axis)
    # calculate the coords of those indices
    bc0 = n.array(n.unravel_index(face0,shape),dtype=dtype)
    bc1 = n.array(n.unravel_index(face1,shape),dtype=dtype)
    # calculate itp, itertools product [-1,0,1]
    itp = fiso.calc_itp(3,True,dtype)
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
    bcn0 = n.ravel_multi_index(nc0,shape,mode=boundary_mode)
    bcn1 = n.ravel_multi_index(nc1,shape,mode=boundary_mode)
    return face0,bcn0,face1,bcn1

fiso.setup = setup
find = fiso.find
