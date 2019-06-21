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
from multiprocessing import Event,Pool

from fiso import fiso
# printing extra diagnostic messages
verbose = True
# how to determine neighbors of boundary cells
timer = fiso.timer
setup = fiso.setup

def find(data):
    #mfw,order,cutoff,pcn = setup(data,'')
    pcn,mfw,bi,bpcn,displacements = fiso.setup_mem(data)
    pcn_parts = [bi,bpcn,displacements]
    flat_data = data.reshape(-1)
    labels = -n.ones(len(flat_data),dtype=int) #indices are real index locations
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
    ready_set = set(mfw)
    nproc = 4
    event = Event()
    space = [None]*nproc
    jobs = deque([])
    
    # initialize
    t0 = time.time()
    climbing = True
    def update_data(walkerresult):
        root_cell,members,nextmembers,lastcell,working,work_set,ready = walkerresult
        iso_dict[root_cell] = members
        working_dict[root_cell] = working
        work_set_dict[root_cell] = work_set
        labels[list(iso_dict[root_cell])] = root_cell
        if lastcell > -1:
            labels[list(nextmembers)] = lastcell
            parent_dict[lastcell] = lastcell
            for child in recursive_child(iso_list,eic_list,root_cell):
                parent_dict[child] = lastcell
            parent_dict[root_cell] = lastcell
            if lastcell in iso_list:
                # add child to pre-existing critical point 
                ili = iso_list.index(lastcell)
                eic_list[ili].append(root_cell)
            else:
                # add a new critical point 
                iso_list.append(lastcell)
                eic_list.append([root_cell])
                heapq.heappush(val_heap,(flat_data[lastcell],iso_list.index(lastcell)))
            if ready:
                ready_set.add(root_cell)
        event.set()
        space.append(None)
        return walkerresult

    pool = Pool(nproc)
    
    while len(iso_dict.keys()) < len(iso_list)-1:
        '''
        for iso in set(iso_list).difference(set(iso_dict.keys())):
            # these need to be done
            if iso not in ready_set:
                parent_of_lessers = set([parent_dict[label] for label in labels[pcn[iso][flat_data[pcn[iso]] < flat_data[iso]]]])
                if -1 not in parent_of_lessers:
                    # this one is ready
                    ready_set.add(iso)
        '''
        iso_i = heapq.heappop(val_heap)[1] # the order_i'th smallest

        mfwi = iso_list[iso_i]
        eics  = eic_list[iso_i]

        while mfwi not in ready_set:
            if len(jobs) > 0:
                event.clear()
                update_data(jobs.popleft().get())
                event.wait()
            else:
                ready_set.add(mfwi)
                
        print(len(val_heap),flat_data[mfwi])
        if labels[mfwi] > -1:
            # this is a "local minimum" which has been explored by equalwalker
            continue
        working = []
        work_set = set()
        for eic in eics:
            working += working_dict.pop(eic)
            work_set.update(work_set_dict.pop(eic))
        heapq.heapify(working)
        parent_dict[mfwi] = mfwi
        space.pop()
        jobs.append(pool.apply_async(walker,args=(flat_data,pcn_parts,parent_dict,labels,mfwi,working,work_set,)))
        #jobs.append(walker(flat_data,pcn_parts,parent_dict,labels,mfwi,working,work_set))
        if len(space) == 0:
            event.clear()
            update_data(jobs.popleft().get())
            event.wait()

        
        
    t1 = time.time()
    print(t1-t0)
    return iso_dict,labels,iso_list,eic_list

def walker(flat_data,pcn_parts,parent_dict,labels,index0,working,work_set):
    bi,bpcn,displacements = pcn_parts
    class pcnDict(dict):
        def __getitem__(self,index):
            return self.get(index,index+displacements)
    pcn = pcnDict(zip(bi,bpcn))
    # working set of cells, larger than members
    t0 = time.time()
    labels[index0] = index0

    for cell in set(pcn[index0][labels[pcn[index0]] == -1]):
        if cell is not index0:
            newitem = (flat_data[cell],cell)
            heapq.heappush(working,newitem)
            work_set.add(cell)
    members = deque([index0]) # cells already added to index0
    nextmembers = deque([])
    growing = True
    lastcell = -1
    while growing:
        # try to grow

        cval,cell = working[0]
        pcnc = pcn[cell]

        # process equal cells
        if cval in flat_data[pcnc]:
            last_cell, ready, equalmembers = equalwalker(flat_data,pcn,parent_dict,labels,cell,working,work_set)
            heapq.heappop(working)
            if last_cell > -1:
                # equalmembers belong to new iso cell
                nextmembers.extend(equalmembers)
                growing = False
            else:
                # equalmembers belong to current working cell
                labels[equalmembers] = index0
                members.extend(equalmembers)
            continue

        # process lesser cells
        lesser = flat_data[pcnc] < cval        
        lpc = set(labels[pcnc[lesser]])
        plpc = list(set([parent_dict[lpci] for lpci in lpc]))

        
        # parents of labels of lesser neighbors of cell
        if len(plpc) == 1:
            # if there is only 1, inherit plpc[0]
            members.append(cell)
            labels[cell] = index0

            # add larger neighbors of cell to working queue
            #addlist = list(set(pcnc[n.logical_not(lesser)]).difference(work_set)) # this is actually slower
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
    return [index0,members,nextmembers,lastcell,working,work_set,ready]


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
        # add entire region (moved to walker)
        # labels[list(members)] = plc[0]
        
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
        # moved to walker
        # labels[list(members)] = index0
        # moved to update_data
        # parent_dict[index0] = index0
        last_cell = index0
        if -1 in plc:
            ready = False
    return last_cell, ready, list(members)

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
