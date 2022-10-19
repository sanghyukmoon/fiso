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
from multiprocessing import Pool,Manager,Event




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
    manager = Manager()
    parent_dict = {}#manager.dict()#{}
    parent_dict[-1] = -1
    iso_dict = {}#manager.dict()#{}
    iso_list = list(mfw)#manager.list(mfw)
    eic_list = list([[]]*len(iso_list)) #manager
    val_heap = zip(flat_data[mfw],range(len(iso_list)))
    heaped_set = set(mfw)
    heapq.heapify(val_heap)
    work_set_dict = dict()#manager.dict()
    working_dict = dict()#manager.dict()
    # ready_dict = dict(zip(mfw,[True]*len(mfw)))
    # initialize
    t0 = time.time()
    climbing = True

    nproc = 4
    event = Event()
    space = [None]*nproc
    def update_data(walkerresult):
        mfwi,members,lastcell,out_working,out_work_set,ready = walkerresult
        iso_dict[mfwi] = members
        working_dict[mfwi] = out_working
        work_set_dict[mfwi] = out_work_set
        labels[list(iso_dict[mfwi])] = mfwi
        if lastcell > -1:
            # ready_dict[lastcell] = ready
            for child in recursive_child(iso_list,eic_list,mfwi):
                parent_dict[child] = lastcell
            parent_dict[mfwi] = lastcell
            if lastcell in iso_list:
                # add child to pre-existing critical point 
                ili = iso_list.index(lastcell)
                eic_list[ili].append(mfwi)
            else:
                print(mfwi,lastcell)
                # add a new critical point 
                iso_list.append(lastcell)
                eic_list.append([mfwi])
            
            if ready:
                if lastcell not in heaped_set:
                    heaped_set.add(lastcell)
                    heapq.heappush(val_heap,(flat_data[lastcell],iso_list.index(lastcell)))

        #print('event set')
        event.set()
        space.append(None)
        return walkerresult

    pool = Pool(nproc)


    # MAIN LOOP
    while ((len(val_heap) > 1) or (len(space) < nproc)):

        # check for ready 
        for iso in set(iso_list).difference(set(iso_dict.keys())).difference(heaped_set):
            # these need to be done
            parent_of_lessers = set([parent_dict[label] for label in set(labels[pcn[iso][flat_data[pcn[iso]] < flat_data[iso]]]) ])
            if parent_of_lessers.issubset(set(eic_list[iso_list.index(iso)])):
                # this one is ready
                
                newitem = (flat_data[iso],iso_list.index(iso))
                heapq.heappush(val_heap,newitem)
                heaped_set.add(iso)

        # when only 1 is on the heap 
        if len(val_heap) == 1:
            # is there a running job?
            if len(space) < nproc:
                # wait for job to finish 
                event.wait()
                if len(space) == 0:
                    event.clear()            
                continue
            # are there others to work on?
            elif len(iso_list) > len(iso_dict):
                print('others')
                newitem = sorted([(flat_data[iso],iso_list.index(iso)) for iso in set(iso_list).difference(set(iso_dict.keys()))])[0]
                newiso = iso_list[newitem[1]]
                if newiso not in heaped_set:
                    heapq.heappush(val_heap,newitem)
                    heaped_set.add(newiso)
                continue
            else:
                break

        iso_i = heapq.heappop(val_heap)[1] # the order_i'th smallest
        mfwi = iso_list[iso_i]
        eics  = eic_list[iso_i]
        print(len(val_heap),flat_data[mfwi],mfwi)
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
        #job = pool.apply_async(walker,args=(flat_data,pcn_parts,parent_dict,labels,mfwi,working,work_set,))
        # take up a space 
        space.pop()
        # if all spaces are taken, force us to wait for something to open
        if len(space) == 0:
            event.clear()
        job = pool.apply_async(walker,args=(flat_data,pcn_parts,parent_dict,labels,mfwi,working,work_set,),callback=update_data)
        # callback triggers event and adds a space when it finishes
        event.wait()

        #walkerresult = job.get()
        # walkerresult = walker(flat_data,pcn,parent_dict,labels,mfwi,working,work_set)
        #update_data(walkerresult)

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
    return [index0,members,lastcell,working,work_set,ready]


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
