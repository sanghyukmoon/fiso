import numpy as n
import scipy.ndimage as sn
import time
from itertools import islice
import itertools
from collections import deque

import fiso

# printing extra diagnostic messages
verbose = True
# how to determine neighbors of boundary cells
timer = fiso.timer
setup = fiso.setup
subsume = fiso.subsume

def find(data,cut=''):
    core_dict,labels,active_cores,order,cutoff,pcn = setup(data,cut)

    # TS: tree specific
    parent_dict = {}
    child_dict = {}
    for core in active_cores:
        parent_dict[core] = core
        child_dict[core] = deque([core])
    # TS: tree specific

    indices = iter(range(cutoff))
    for i in indices:
        orderi = order[i]
        nls = n.array(list(set(labels[
            pcn[orderi]
        ])))
        nls0 = nls[nls >= 0]
        nnc = len(nls0)
        # number of neighbors in cores
        # first note this cell has been explored
        labels[orderi] = -2
        if (nnc > 0):
            if -2 in nls:
                # a neighbor is previously explored but not cored (boundary), deactivate cores
                collide(active_cores,nls0)
                if len(active_cores) == 0:
                    next(islice(indices,cutoff-i-1,cutoff-i-1),None)
                continue

            # TS
            parents = list(set([parent_dict[nlsi] for nlsi in nls0]))
            npc = len(parents)
            # TS
            
            if (npc == 1):
                # only 1 neighbor, inherit
                inherit = parents[0]
                if inherit in active_cores:
                    labels[orderi] = inherit
                    core_dict[inherit].append(orderi)
                    # inherit from neighbor, only 1 is positive/max
                continue
            elif (npc == 2):
                if set(nls0) <= set(active_cores):
                    # check smaller neighbor if it is too small
                    l0 = len(core_dict[nls0[0]])
                    l1 = len(core_dict[nls0[1]])
                    if min(l0,l1) < 27:
                        subsume(l0,l1,orderi,nls0,core_dict,labels,active_cores)
                        continue
            # There are 2 or more large neighbors to deactivate
            # corei is real index
            # TS
            merge(orderi,active_cores,core_dict,child_dict,parent_dict,parents)
            # TS
            # collide(active_cores,nls0)
            if verbose:
                print(i,' of ',cutoff,' cells ',
                      len(active_cores),' minima')
            if len(active_cores) == 0:
                next(islice(indices,cutoff-i-1,cutoff-i-1),None)
                # skip up to next core or end
        else:
            # no lesser neighbors
            if orderi in active_cores:
                labels[orderi] = orderi
    dt = timer('loop finished for ' + str(cutoff) + ' items')
    if verbose: print(str(dt/i) + ' per cell')
    if verbose: print(str(dt/cutoff) + ' per total cell')
    return core_dict,labels

#if alone, start new core
#if neighbor is in an active core, add to core
#if neighor is in an inactive core, don't add to core
#if 2 or more neighbors are in different cores, dont add to a core.

def collide(active_cores,nls0):
    for nlsi in nls0:
        if nlsi in active_cores:
            active_cores.remove(nlsi)


def merge(orderi,active_cores,core_dict,child_dict,parent_dict,parents):
    merge_bool = False
    for iso in parents:
        if iso in active_cores:
            merge_bool = True
        else:
            print('Unexpected error: inactive parent')
    if merge_bool:
        # start new core, assign it to be its own child and parent
        core_dict[orderi] = deque([orderi])
        child_dict[orderi] = deque([orderi])
        parent_dict[orderi] = orderi
        active_cores.append(orderi) 
    for iso in parents:
        if iso in active_cores:
            # new parent orderi owns all children of all merging isos
            # children's parent is orderi.
            # orderi's children is children
            for child in child_dict[iso]:
                parent_dict[child] = orderi
            child_dict[orderi] += child_dict[iso]
            # deactivate iso
            active_cores.remove(iso)
