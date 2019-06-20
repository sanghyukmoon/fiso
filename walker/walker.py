# This extension returns extra objects
# iso_list: for keeping track of all members of the tree
# eic_list: for keeping track of the tree structure, the immediate children
# of each member

import numpy as n
import scipy.ndimage as sn
import time
from itertools import islice
from collections import deque
import heapq
#from numba import jit

from fiso import fiso
# printing extra diagnostic messages
verbose = True
# how to determine neighbors of boundary cells
timer = fiso.timer
setup = fiso.setup

def walkers(data):
    mfw,order,cutoff,pcn = setup(data,'')
    flat_data = data.reshape(-1)
    labels = -n.ones(len(order),dtype=int) #indices are real index locations
    # set data structures
    parent_dict = {}
    parent_dict[-1] = -1
    iso_dict = {}
    iso_list = list(mfw)
    eic_list = [[]]*len(iso_list)
    val_heap = zip(flat_data[mfw],range(len(iso_list)))
    heapq.heapify(val_heap)
    work_set_dict = {}
    working_dict = {}
    ready_dict = dict(zip(mfw,[True]*len(mfw)))
    # initialize
    t0 = time.time()
    climbing = True
    while len(val_heap) > 1:
        iso_i = heapq.heappop(val_heap)[1] # the order_i'th smallest
        mfwi = iso_list[iso_i]
        eics  = eic_list[iso_i]
        print(mfwi,eics,flat_data[mfwi])
        if label[mfwi] > -1:
            # this is a "local minimum" which has been explored by equalwalker
            continue
        working = []
        work_set = set()
        for eic in eics:
            working += working_dict.pop(eic)
            work_set.update(work_set_dict.pop(eic))
        heapq.heapify(working)

        iso_dict[mfwi],lastcell,working,work_set,ready = walker(flat_data,pcn,parent_dict,labels,mfwi,working,work_set)
        working_dict[mfwi] = working
        work_set_dict[mfwi] = work_set
        if lastcell > -1:
            ready_dict[lastcell] = ready
            for child in recursive_child(iso_list,eic_list,mfwi):
                parent_dict[child] = lastcell
            parent_dict[mfwi] = lastcell
            if lastcell in iso_list:
                # add child to pre-existing critical point 
                ili = iso_list.index(lastcell)
                eic_list[ili].append(mfwi)
            else:
                # add a new critical point 
                iso_list.append(lastcell)
                heapq.heappush(val_heap,(flat_data[lastcell],len(iso_list)-1))
                eic_list.append([mfwi])
        climbing = len(val_heap) > 1

    t1 = time.time()
    print(t1-t0)
    return iso_dict,labels

#@jit
def walker(flat_data,pcn,parent_dict,labels,index0,working,work_set):
    # working set of cells, larger than members
    t0 = time.time()
    parent_dict[index0] = index0
    labels[index0] = index0

    for cell in set(pcn[index0][labels[pcn[index0]] == -1]):
        if cell is not index0:
            newitem = (flat_data[cell],cell)
            heapq.heappush(working,newitem)
            work_set.add(cell)
    members = deque([index0]) # cells already added to index0
    growing = True
    lastcell = -1
    while growing:
        # try to grow

        cval,cell = working[0]
        pcnc = pcn[cell]
        lesser = flat_data[pcnc] < cval

        if cval in flat_data[pcnc]:
            last_cell, ready = equalwalker(flat_data,pcn,parent_dict,labels,cell,working,work_set)
            heapq.heappop(working)
            if last_cell > -1:
                growing = False
            continue

        
        lpc = set(labels[pcnc[lesser]])
        plpc = list(set([parent_dict[lpci] for lpci in lpc]))

        
        # parents of labels of lesser neighbors of cell
        if len(plpc) == 1:
            # if there is only 1, inherit plpc[0]
            members.append(cell)
            labels[cell] = index0

            # add larger neighbors of cell to working queue
            addlist = [thing for thing in pcnc[n.logical_not(lesser)] if thing not in work_set]
            if len(addlist) > 0:
                addlist0 = (flat_data[addlist[0]],addlist[0])
                for celli in addlist[1:]:
                    heapq.heappush(working,(flat_data[celli],celli))
                    work_set.add(celli)
                heapq.heapreplace(working,addlist0)
                work_set.add(addlist[0])
            else:
                heapq.heappop(working)
        else:
            if -1 in plpc:
                ready = False
            else:
                ready = True
            growing = False
            lastcell = cell
    # at the end, sort members by data 
    return members,lastcell,working,work_set,ready

def equalwalker(flat_data,pcn,parent_dict,labels,index0,working,work_set):
    # working set of cells, larger than members
    val0 = flat_data[index0]
    equeue = deque([index0])
    members = deque([])
    touch_set = set()
    lesser_set = set()
    greater_set = set()
    growing = True
    while growing:
        cell = equeue.pop()
        # add to members list
        members.append(cell)
        touch_set.add(cell)
        pcnc = pcn[cell]
        fpcnc = flat_data[pcnc]
        # get equal neighbors 
        for celli in pcnc[fpcnc == val0]:
            if celli not in touch_set:
                touch_set.add(celli)
                equeue.append(celli)
        lesser_set.update(set(pcnc[fpcnc < val0]))
        greater_set.update(set(pcnc[fpcnc > val0]))
        growing = (len(equeue) > 0)
    # use lesser_set to determine who this belongs to
    # use greater_set to add to working list 
    plc = list(set([parent_dict[label] for label in set(labels[list(lesser_set)])]))
    # parents of lessers to cell
    ready = True
    if len(plc) == 1:
        # add entire region
        labels[list(members)] = plc[0]
        
        # prevents members from being worked on in future 
        work_set.update(set(members))

        # add greater cells to work set
        for celli in greater_set:
            if celli not in work_set:
                heapq.heappush(working,(flat_data[celli],celli))
                work_set.add(celli)
        last_cell = -1
    else:
        # start new region
        labels[list(members)] = index0
        parent_dict[index0] = index0
        last_cell = index0
        if -1 in plc:
            ready = False
        # wish I could do this:
        #for child in recursive_child(iso_list,eic_list,mfwi):
        #    parent_dict[child] = lastcell
    return last_cell, ready









def find(data,cut=''):
    mfw,order,cutoff,pcn = setup(data,cut)
    #iso dict and labels setup
    iso_dict = {}
    labels = -n.ones(len(order),dtype=int) #indices are real index locations
    #inside loop, labels are accessed by labels[order[i]]
    for mini in mfw:
        iso_dict[mini] = deque([mini])
    labels[mfw] = mfw
    active_isos = set(mfw) #real index

    # TS: tree specific
    iso_list = []
    eic_list = [] #exclusive immediate children
    parent_dict = {}
    child_dict = {}
    for iso in active_isos:
        parent_dict[iso] = iso
        child_dict[iso] = deque([iso])
        iso_list.append(iso)
        eic_list.append([])
    # TS: tree specific

    indices = iter(range(cutoff))

    min_active = 1
    for i in indices:
        orderi = order[i]
        nls = set(labels[
            pcn[orderi]
        ])
        nls0 = nls.copy()
        nls0.discard(-1)
        nls0.discard(-2)
        nls0 = list(set(
            [parent_dict[nlsi] for nlsi in nls0]
        ))
        nnc = len(nls0)
        # number of neighbors in isos
        # first note this cell has been explored
        labels[orderi] = -2
        if (nnc > 0):
            if -2 in nls:
                # a neighbor is previously explored but not isod (boundary), deactivate isos
                collide(active_isos,nls0)
                min_active = 0
                if len(active_isos) == min_active:
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
            # There are 2 or more neighbors to deal with
            # TS
            merge(active_isos,nls0,orderi,iso_dict,child_dict,parent_dict,iso_list,eic_list,labels)
            # TS
            if verbose:
                print(i,' of ',cutoff,' cells ',
                      len(active_isos),' minima')
            if len(active_isos) == min_active:
                next(islice(indices,cutoff-i-1,cutoff-i-1),None)
                # skip up to next iso or end
        else:
            # no lesser neighbors
            if orderi in active_isos:
                labels[orderi] = orderi
    dt = timer('loop finished for ' + str(cutoff) + ' items')
    if verbose: print(str(dt/i) + ' per cell')
    if verbose: print(str(dt/cutoff) + ' per total cell')
    return iso_dict,labels,iso_list,eic_list

#if alone, start new iso
#if neighbor is in an active iso, add to iso
#if neighor is in an inactive iso, don't add to iso
#if 2 or more neighbors are in different isos, dont add to a iso.
# Q: What happens when a child's parent is subsumed? Their new parent is the subsumer
# Q: What happens when a child's parent is deactivated in collision with boundary?
# Who is their new parent?
# During merge, parents that get inactivated are replaced with active parent.
# When a child's parent is inactive
# to turn tree into normal, just set merge to collide

def collide(active_isos,nls0,*args):
    for nlsi in nls0:
        if nlsi in active_isos:
            active_isos.discard(nlsi)

def merge(active_isos,parents,orderi,iso_dict,child_dict,parent_dict,iso_list,eic_list,labels):
    # merge removes deactivated parents
    # subsume removes deactivated parents
    # hence, inactive parents must have come from collide
    # first check for inactive parents which must have come from collide
    # if any inactive parents, collide
    # if all parents are active, first subsume smallest isos. 
    if set(parents) <= active_isos:
        tree_subsume(active_isos,parents,orderi,iso_dict,child_dict,parent_dict,iso_list,eic_list,labels)
    else:
        collide(active_isos,parents)
        return 
    eic = list(set(parents) & active_isos)
    # eic = list(parents)
    # start new iso, assign it to be its own child and parent
    iso_dict[orderi] = deque([orderi])
    labels[orderi] = orderi
    child_dict[orderi] = deque([orderi])
    parent_dict[orderi] = orderi

    # new parent iso is active
    active_isos.add(orderi)
    iso_list.append(orderi)
    eic_list.append(eic)
    # exclusive immediate children
    for iso in eic:
        # new parent orderi owns all children of all merging isos
        # orderi's children is children
        child_dict[orderi] += child_dict[iso]
        # children's parent is orderi.
        for child in child_dict[iso]:
            parent_dict[child] = orderi
            # deactivate iso
        active_isos.discard(iso)
    

def tree_subsume(active_isos,parents,orderi,iso_dict,child_dict,parent_dict,iso_list,eic_list,labels):
    min_cells = 27
    eic_dict = dict(zip(iso_list,eic_list))
    subsume_set = set()
    len_list = [None] * len(parents)
    cell_list = [None] * len(parents)
    for i in range(len(parents)):
        parent = parents[i]
        lidp = len(iso_dict[parent])
        if lidp < min_cells:
            # too small, try making bigger
            cell_list[i] = recursive_members(iso_dict,eic_dict,parent)
            lidp = len(cell_list[i])
            if lidp < min_cells:
                # still too small 
                subsume_set.add(i)
        else: # big enough
            cell_list[i] = iso_dict[parent]
        len_list[i] = lidp
    largest_i = n.argmax(len_list)
    largest_parent = parents[largest_i]
    subsume_set.discard(largest_i)
    # add too small to largest
    for i in subsume_set:
        # add smaller iso cells to larger dict
        iso_dict[largest_parent] += cell_list[i]
        # relabel smaller iso cells to larger
        labels[cell_list[i]] = largest_parent
        # delete smaller iso
        active_isos.remove(parents[i])
        iso_dict.pop(parents[i])
        # note any children of smaller iso would be too small 
        # and be previously subsumed
    
# Keep ordered list of isos [ ]
# Keep ordered list of lists [ ]. Each list is immediate children.
# When considering merges in post, move along ordered list of isos
# and resolve from bottom to top. 1 pass.
# Each iso ixs processed before its parents.

def recursive_num_members(iso_dict,eic_dict,iso):
    output = len(iso_dict[iso])
    for child_iso in eic_dict[iso]:
        output += recursive_num_members(iso_dict,eic_dict,child_iso)
    return output

def recursive_members(iso_dict,eic_dict,iso):
    output = []
    output += iso_dict[iso]
    for child_iso in eic_dict[iso]:
        output += recursive_members(iso_dict,eic_dict,child_iso)
    return output

def recursive_child(iso_list,eic_list,iso):
    output = []
    children = eic_list[iso_list.index(iso)]
    output += children
    for child_iso in children:
        output += recursive_child(iso_list,eic_list,child_iso)
    return output
