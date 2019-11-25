import numpy as n
import scipy.ndimage as sn
import time
from itertools import islice
import itertools
from collections import deque

# printing extra diagnostic messages
verbose = True
# how to determine neighbors of boundary cells
boundary_mode = 'clip'
# whether diagonal cells are neighbors
corner_bool = True
time.prevtime = time.time()

def timer(string=''):
    thistime = time.time()
    dt = thistime - time.prevtime
    time.prevtime = thistime
    if len(string) > 0:
        if verbose: print(string,str(dt) + " seconds elapsed")
    return dt

def setup(data,cut):
    timer()
    #prepare data
    dshape = data.shape
    dlist = data.reshape(-1) #COPY
    order = dlist.argsort() #an array of real index locations #COPY
    timer('sort')
    cutoff = len(order) #number of cells to process
    #optional cutoff
    if type(cut) is float:
        cutoff = n.searchsorted(dlist[order],cut)

    #precompute neighbor indices
    pcn = precompute_neighbor(dshape,corner=corner_bool,mode=boundary_mode)
    timer('precompute neighbor indices')

    #timer('init short')
    minima_flat = find_minima_flat(data)
    #indices of the minima in original
    mfw = n.where(minima_flat)[0]
    #mfw is real index
    if verbose: print(len(mfw),'minima')
    return mfw,order,cutoff,pcn

def find(data,cut=''):
    # take in nd data
    # find iso
    # setup
    mfw,order,cutoff,pcn = setup(data,cut)
    #iso dict and labels setup
    iso_dict = {}
    labels = -n.ones(len(order),dtype=int) #indices are real index locations
    #inside loop, labels are accessed by labels[order[i]]
    for mini in mfw:
        iso_dict[mini] = deque([mini])
    labels[mfw] = mfw
    active_isos = set(mfw) #real index

    # loop
    indices = iter(range(cutoff))
    # note indices = iter(xrange(cutoff)) is ~1% faster loop in python2.7
    # note looping over i and getting order[i] is only
    # 1% slower than iter over order
    # added convenience of smooth exit from loop
    for i in indices:
        orderi = order[i]
        # grab unique neighbor labels
        # nli = pcn[:,orderi]
        nls = set(labels[
            pcn[orderi]
        ])
        nls0 = nls.copy()
        nls0.discard(-1)
        nls0.discard(-2)
        nls0 = list(nls0)
        # nls = n.unique(labels[nli]) #this is much slower
        # nnc = (nls >= 0).sum() #this is 2x slower
        nnc = len(nls0)
        # number of neighbors in isos

        # first note this cell has been explored
        labels[orderi] = -2
        if (nnc > 0):
            if -2 in nls:
                # a neighbor is previously explored but not isod (boundary), deactivate isos
                collide(active_isos,nls0)
                if len(active_isos) == 0:
                    next(islice(indices,cutoff-i-1,cutoff-i-1),None)
                continue
            if (nnc == 1):
                # only 1 neighbor, inherit
                inherit = nls0[0]
                if inherit in active_isos:
                    labels[orderi] = inherit
                    iso_dict[inherit].append(orderi)
                    # inherit from neighbor, only 1 is positive/max
                continue
            elif (nnc == 2):
                if set(nls0) <= active_isos:
                    # check smaller neighbor if it is too small
                    l0 = len(iso_dict[nls0[0]])
                    l1 = len(iso_dict[nls0[1]])
                    if min(l0,l1) < 27:
                        subsume(l0,l1,orderi,nls0,iso_dict,labels,active_isos)
                        continue
            # There are 2 or more large neighbors to deactivate
            # isoi is real index
            collide(active_isos,nls0)
            if verbose:
                print(i,' of ',cutoff,' cells ',
                      len(active_isos),' minima')
            if len(active_isos) == 0:
                next(islice(indices,cutoff-i-1,cutoff-i-1),None)
                # skip up to next iso or end
        else:
            # no lesser neighbors
            if orderi in active_isos:
                labels[orderi] = orderi
    dt = timer('loop finished for ' + str(cutoff) + ' items')
    if verbose: print(str(dt/i) + ' per cell')
    if verbose: print(str(dt/cutoff) + ' per total cell')
    return iso_dict,labels

#if alone, start new iso
#if neighbor is in an active iso, add to iso
#if neighor is in an inactive iso, don't add to iso
#if 2 or more neighbors are in different isos, dont add to a iso.

def find_minima_no_bc(arr):
    '''
    Find minima using sn, don't allow any boundary cells to be minima
    Then add the boundaries
    '''
    if corner_bool:
        nhbd = sn.generate_binary_structure(len(arr.shape),3)
    else:
        nhbd = sn.generate_binary_structure(len(arr.shape),1)
    # nhbd[len(arr.shape)*[slice(1,2)]] = False #exclude self
    mode0 = 'constant'
    # initially doesnt allow any boundary cells to be minima
    local_min = (arr == sn.filters.minimum_filter(arr,
                                                  footprint=nhbd,
                                                  mode=mode0,
                                                  cval=-n.inf)).reshape(-1)
    return local_min

def find_minima_boundary_only(dlist,indices,bcn):
    '''
    dlist: 1-d array data
    indices: 1-d array of boundary flattened indices
    bcn: boundary neighbors: num_indices x num_neighbors 2-d array of
    flattened indices
    output: indices that are local minima.
    '''
    indices = n.array(indices)
    return indices[dlist[indices] <= n.min(dlist[bcn],axis=1)]


def find_minima_global(arr):
    # find minima function depending on global args
    if corner_bool:
        nhbd = sn.generate_binary_structure(len(arr.shape),3)
    else:
        nhbd = sn.generate_binary_structure(len(arr.shape),1)
    if boundary_mode == 'clip':
        mode0 = 'reflect'
    elif boundary_mode == 'wrap':
        mode0 = 'wrap'
    else:
        mode0 = 'reflect'
    # nhbd[len(arr.shape)*[slice(1,2)]] = False #exclude self, enforce strict local minimum
    # local_min = (arr < sn.filters.minimum_filter(arr,
    local_min = (arr == sn.filters.minimum_filter(arr,
                                                 footprint=nhbd,
                                                 mode=mode0))
    return local_min

find_minima = find_minima_global

def find_minima_flat(arr):
    return find_minima(arr).reshape(-1)

def find_minima_pcn(dlist,pcn):
    return (dlist <= n.min(dlist[pcn],axis=1))
    # go from flattened array and pcn to flat
    # compare each cell to its neighbors according to pcn
    # method is 10x slower than sn.minimum_filter but more general

def boundary_pcn(coords,itp,shape,corner,mode='clip'):
    newcoords = coords[:,:,None] + n.transpose(itp)[:,None,:]
    output = n.ravel_multi_index(newcoords,shape,mode=mode)
    return output
    # start with coords of shape dim,num_coords
    # for each num_coords, add one of the many itp
    # itp has shape num_neighbors,dim
    # n.transpose(itp)
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
        dni[i] = n.arange(shape[i],dtype=dtype)
    #for boundary dimensions i set indices j != i, setting index i to be
    #0 or end

    ndnis = dim*[None]
    i = axis
    shapei = shape[:] #copy shape
    shapei[i] = 1     #set dimension i to 1 (flat boundary)
    nzs = n.zeros(shapei,dtype=dtype) #initialize boundary to 0
    for j in idx:
        if j == i:
            continue
        #make coord j using the n.arange (dni) with nzs of desired shape
        selj = basel[:]
        selj[j] = slice(None)
        #slicing on index j makes dni[j] vary on index j and copy on other dimensions with desired shape nzs
        ndnis[j] = dni[j][tuple(selj)] + nzs
    ndnis[i] = 0
    face0 = list(n.ravel_multi_index(ndnis,shape).reshape(-1))
    ndnis[i] = shape[i]-1
    face1 = list(n.ravel_multi_index(ndnis,shape).reshape(-1))
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
        dni[i] = n.arange(shape[i],dtype=dtype)

    #for boundary dimensions i set indices j != i, setting index i to be
    #0 or end
    for i in idx:

        ndnis = ls*[None]
        shapei = shape[:] #copy shape
        shapei[i] = 1     #set dimension i to 1 (flat boundary)
        nzs = n.zeros(shapei,dtype=dtype) #initialize boundary to 0
        for j in idx:
            if j == i:
                continue
                #make coord j using the n.arange (dni) with nzs of desired shape
            selj = basel[:]
            selj[j] = slice(None)
            #slicing on index j makes dni[j] vary on index j and copy on other dimensions with desired shape nzs
            ndnis[j] = dni[j][tuple(selj)] + nzs
        ndnis[i] = 0
        bi += list(n.ravel_multi_index(ndnis,shape).reshape(-1))
        ndnis[i] = shape[i]-1
        bi += list(n.ravel_multi_index(ndnis,shape).reshape(-1))
    return bi

def calc_itp(dim,corner,dtype):
    # Compute 1-D flattened array offsets corresponding to neighbors 
    offs = [-1,0,1]
    itp = list(itertools.product(offs,repeat=dim))
    if corner:
        itp.remove((0,)*dim)
    else:
        itp = [i for i in itp if i.count(0) == 2]
    itp = n.array(itp,dtype=dtype)
    return itp

def compute_displacement(shape,corner):
    nps = n.prod(shape)
    #save on memory when applicable
    if nps < 2**31:
        dtype = n.int32
    else:
        dtype = n.int64
    dim = len(shape)
    itp = calc_itp(dim,corner,dtype)
    ishape = shape[::-1]
    factor = n.cumprod(n.append([1],shape[::-1]))[:-1][::-1]
    factor = factor.astype(dtype)
    displacements = (itp*factor).sum(axis=1)
    return displacements

def precompute_neighbor(shape,corner=True,mode='clip'):
    nps = n.prod(shape)
    #save on memory when applicable
    if nps < 2**31:
        dtype = n.int32
    else:
        dtype = n.int64
    #set up array of cartesian displacements (itp)
    dim = len(shape)
    itp = calc_itp(dim,corner,dtype)
    #set up displacements in index space (treat n-d array as 1-d list)
    ishape = shape[::-1]
    factor = n.cumprod(n.append([1],shape[::-1]))[:-1][::-1]
    factor = factor.astype(dtype)
    displacements = (itp*factor).sum(axis=1)
    #displacements is num_neighbors 1-d array
    indices = n.arange(nps,dtype=dtype)[:,None]
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
    boundary_coords = n.array(n.unravel_index(boundary_indices,shape),
                              dtype=dtype)
    bpcn = boundary_pcn(boundary_coords,itp,shape,corner,mode=mode).astype(dtype)
    return boundary_indices,bpcn

def collide(active_isos,nls0):
    for nlsi in nls0:
        if nlsi in active_isos:
            active_isos.remove(nlsi)

def subsume(l0,l1,orderi,nls0,iso_dict,labels,active_isos):
    smaller = n.argmin([l0,l1])
    larger = 1-smaller
    #add smaller iso cells to larger dict
    iso_dict[nls0[larger]] += iso_dict[nls0[smaller]]
    #relabel smaller iso cells to larger
    labels[tuple(iso_dict[nls0[smaller]])] = nls0[larger]
    active_isos.remove(nls0[smaller])
    iso_dict.pop(nls0[smaller])

    labels[orderi] = nls0[larger]
    iso_dict[nls0[larger]].append(orderi)


def setup_mem(data,corner=corner_bool):
    shape = data.shape
    dlist = data.reshape(-1)
    nps = n.prod(shape)
    #save on memory when applicable
    if nps < 2**31:
        dtype = n.int32
    else:
        dtype = n.int64
    #set up array of cartesian displacements (itp)
    dim = len(shape)
    itp = calc_itp(dim,corner,dtype)
    bi,bpcn = boundary_i_bcn(shape,dtype,itp,corner,boundary_mode)
    displacements = compute_displacement(shape,corner)
    class pcnDict(dict):
        def __getitem__(self,index):
            return self.get(index,index+displacements)
    pcn = pcnDict(zip(bi,bpcn))
    mfw0 = n.where(find_minima_no_bc(data).reshape(-1))[0]
    mfw1 = find_minima_boundary_only(dlist,bi,bpcn)
    mfw = n.unique(n.sort(n.append(mfw0,mfw1))) 
    return pcn,mfw,bi,bpcn,displacements
