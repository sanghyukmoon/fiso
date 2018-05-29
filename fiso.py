import numpy as n
import scipy.ndimage as sn
import time
from itertools import islice
import itertools
from collections import deque

verbose = True
time.prevtime = time.time()

def timer(string=''):
    thistime = time.time()
    dt = thistime - time.prevtime
    time.prevtime = thistime
    if len(string) > 0:
        if verbose: print(string,str(dt) + " seconds elapsed")
    return dt

def find(data,cut=''):
    #take in 3d data
    #find iso
    timer()

    #prepare data
    dshape = data.shape
    dlist = data.reshape(-1) #COPY
    order = dlist.argsort() #an array of real index locations #COPY
    dmin = dlist[order[0]]
    dmax = dlist[order[-1]]

    timer('sort')
    length = len(order)
    cutoff = length #number of cells to process
    ## max_disc = 2.0*(dmax - dmin) / float(min(dshape)) #DE

    #optional cutoff
    if type(cut) is float:
        cutoff = n.searchsorted(dlist[order],cut)
    timer('init short')
    minima_flat = find_minima_flat(data,dmin)
    #indices of the minima in original
    mfw = n.where(minima_flat)[0]
    sorted_mfw = mfw[n.argsort(dlist[mfw])] #sorted by phi
    #i need sorted indices of mfw to make leaps
    ##num_mins = len(mfw) #DE
    #sort minima in order
    #mfw is real index
    # if verbose: print(num_mins,'minima')
    if verbose: print(len(mfw),'minima')

    #core dict and labels setup
    core_dict = {}
    labels = -n.ones(length,dtype=int) #indices are real index locations
    #inside loop, labels are accessed by labels[order[i]]
    for mini in mfw:
        core_dict[mini] = deque([mini])
    labels[mfw] = mfw
    active_cores = list(sorted_mfw) #real index
    inactive_cores = [] #real index
    timer('init minima')

    #precompute neighbor indices 
    pcn = precompute_neighbor3(dshape)

    #loop
    indices = iter(range(cutoff))
    for i in indices:
        #grab unique neighbor labels
        nli = pcn[:,order[i]]
        nls = n.array(list(set(labels[nli])))
        #nls = n.unique(labels[nli]) #this is much slower
        #nnc = (nls >= 0).sum() #this is 2x slower
        nls0 = nls[nls >= 0]        
        nnc = len(nls0)
        #number of neighbors in cores
        #first note this cell has been explored

        if labels[order[i]] == -1:
            labels[order[i]] = -2
        else:
            continue
        if (nnc > 0):
            if -2 in nls:
                #a neighbor is previously explored but not cored (boundary), deactivate cores
                for nlsi in nls0:
                    if nlsi in active_cores:
                        active_cores.remove(nlsi)
                        if len(active_cores) == 0:
                            next(islice(indices,cutoff-i-1,cutoff-i-1),None)
                continue
            if (nnc == 1):
                #only 1 neighbor, inherit
                #use nls0[0] instead of max(nls)?
                if max(nls) in active_cores:
                    labels[order[i]] = max(nls) 
                    core_dict[max(nls)].append(order[i])   
#inherit from neighbor, only 1 is positive/max
                continue
            elif (nnc == 2):
                if set(nls0) <= set(active_cores):
                #check smaller neighbor if it is too small
                    l0 = len(core_dict[nls0[0]])
                    l1 = len(core_dict[nls0[1]])
                    if min(l0,l1) < 27:
                        smaller = n.argmin([l0,l1])
                        larger = 1-smaller
                    #add smaller core cells to larger dict
                        core_dict[nls0[larger]] += core_dict[nls0[smaller]]
                    #relabel smaller core cells to larger
                        labels[core_dict[nls0[smaller]]] = nls0[larger]
                        active_cores.remove(nls0[smaller])
                        core_dict.pop(nls0[smaller])

                        labels[order[i]] = nls0[larger] 
                        core_dict[nls0[larger]].append(order[i])
                        continue

            #There are 2 or more large neighbors to deactivate
            ##aclimit = dlist[order[i]] - max_disc #DE
            #ac calculation taken out of loop 
            #corei is real index
            for corei in nls0:
                if corei in active_cores:
                    '''
                    #test to see if cores has gained member recently
                    #remove strong discontinuity
                    current_ac_index = active_cores.index(corei) - 1
                    for aci in range(current_ac_index,-1,-1):
                        if dlist[active_cores[aci]] < aclimit:
                            active_cores.remove(active_cores[aci])
                    #remove current collision
                    '''
                    active_cores.remove(corei)
                    if verbose:
                        print(i,' of ',cutoff,' cells ',
                              len(active_cores),' minima')
                    if len(active_cores) == 0:
                        next(islice(indices,cutoff-i-1,cutoff-i-1),None)
                    #skip up to next core or end
    dt = timer('loop finished for ' + str(cutoff) + ' items')
    if verbose: print(str(dt/cutoff) + ' per item')
    return core_dict,labels

#if alone, start new core
#if neighbor is in an active core, add to core
#if neighor is in an inactive core, don't add to core
#if 2 or more neighbors are in different cores, dont add to a core. 

def find_minima(arr,arrmin):
    #arrmin prevents boundary from being accepted as local minima
    smaller = -2.0*n.abs(arrmin)
    #nhbd = sn.generate_binary_structure(len(arr.shape),1) #neighborhood
    nhbd = sn.generate_binary_structure(len(arr.shape),3) #neighborhood
    #local_min = (sn.filters.minimum_filter(arr, footprint=nhbd,mode='constant',cval=smaller)==arr)
    local_min = (sn.filters.minimum_filter(arr, footprint=nhbd)==arr)
    return local_min

def find_minima_flat(arr,arrmin):
    return find_minima(arr,arrmin).reshape(-1)

def precompute_neighbor(shape):
    coords = n.indices(shape)
    output = range(len(shape)*2)
    index = 0
    for dimi in range(len(shape)):
        for offset in [-1,1]:
            newcoords = coords.copy()
            newcoords[dimi] += offset
            output[index] = n.ravel_multi_index(newcoords,shape,mode='clip').reshape(-1)
            index += 1
    return n.array(output)

def precompute_neighbor2(shape,corner=True):      
    coords = n.indices(shape) 
    return subpcn2(coords,shape,corner=corner)

def subpcn2(coords,shape,corner=True):
    '''given an array shape, find the neighbor indices of given coordinates'''
    lc = len(coords.shape) - 1
    offs = [-1,0,1]   
    itp = list(itertools.product(offs,repeat=lc))     
    if corner:
        itp.remove((0,)*lc)       
    else:
        itp = [i for i in itp if i.count(0) == 2]
    sel = [slice(None)] + [slice(None)] + [None] * lc 
    #2 for going through all itp, duplicate to fit coords [3,:,:,:]
    coordsel = [slice(None)]*(lc+2)   
    #total length lc+2, duplication over neighbors coordsel[1]
    coordsel[1] = None
    newcoords = coords[coordsel] + n.transpose(itp)[sel]      
    output = n.ravel_multi_index(newcoords,shape,mode='clip').reshape(len(itp),-1)    
    return output 

def gbi(shape,dtype):     
    '''get boundary indices from shape'''   
    shape = list(shape)
    bi = []   
    ls = len(shape) 
    basel = ls*[None] #array slice none to extend array dimension
    idx = range(ls) #index for loops
    #dni is the coords for dimension "i"
    dni = range(ls) 
    for i in idx:   
        dni[i] = n.arange(shape[i],dtype=dtype)   

    #for boundary dimensions i set indices j != i, setting index i to be 
    #0 or end
    for i in idx:   

        ndnis = range(ls) 
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
            ndnis[j] = dni[j][selj] + nzs   
        ndnis[i] = 0
        bi += list(n.ravel_multi_index(ndnis,shape).reshape(-1))    
        ndnis[i] = shape[i]-1   
        bi += list(n.ravel_multi_index(ndnis,shape).reshape(-1))    
    return bi

def precompute_neighbor3(shape,corner=True):                         
    nps = n.prod(shape)                                              
    #save on memory when applicable 
    if nps < 2**31:
        dtype = n.int32                                              
    else:                                                            
        dtype = n.int64                            
    #set up array of cartesian displacements (itp)
    lc = len(shape)                                                  
    offs = [-1,0,1]                                                  
    itp = list(itertools.product(offs,repeat=lc))                    
    if corner:                                                       
        itp.remove((0,)*lc)                                          
    else:                                                            
        itp = [i for i in itp if i.count(0) == 2]                    
    #set up displacements in index space (treat n-d array as 1-d list)
    ishape = shape[::-1] 
    factor = n.cumprod(n.append([1],shape[::-1]))[:-1][::-1]  
    factor = factor.astype(dtype)
    itp = n.array(itp,dtype=dtype)
    displacements = (itp*factor).sum(axis=1)                         
    #displacements is num_neighbors 1-d array
    indices = n.arange(nps,dtype=dtype)[None]
    #indices is 1-d array, 1 for each cell
    pcn = indices + displacements[:,None] 
    #pcn is 2-d array using :,None to combine 
    #apply boundary correction mode='clip' set in subpcn2
    boundary_indices = gbi(shape,dtype)                              
    boundary_coords = list(n.array(n.unravel_index(boundary_indices,shape),dtype=dtype))                                                 
    for i in range(lc):                                              
        boundary_coords[i] = boundary_coords[i].reshape(1,1,-1)      
    boundary_coords = n.array(boundary_coords,dtype=dtype)           
    bcn = subpcn2(boundary_coords,shape,corner=True)                 
    #pcn shape num_neighbors x cells
    #bcn shape num_neighbors x cells
    pcn[:,boundary_indices] = bcn                                    
    return pcn          

