# This extension returns extra objects
# iso_list: for keeping track of all members of the tree
# eic_list: for keeping track of the tree structure, the immediate children
# of each member

from collections import deque
from itertools import islice
import numpy as np
from .fiso import setup
from .tools import timer


def find(data, verbose=True):
    # TODO(SMOON) Use more meaningful function name; add docstring
    idx_minima, cells_ordered, cutoff, pcn = setup(data)
    #iso dict and labels setup
    iso_dict = {}
    labels = -np.ones(len(cells_ordered), dtype=int) #indices are real index locations
    #inside loop, labels are accessed by labels[cells_ordered[i]]
    for mini in idx_minima:
        iso_dict[mini] = deque([mini])
    labels[idx_minima] = idx_minima
    active_isos = set(idx_minima) #real index

    iso_list = []
    eic_list = [] #exclusive immediate children
    parent_dict = {}
    child_dict = {}
    for iso in active_isos:
        parent_dict[iso] = iso # The potential minimum "parents" the cells contained in the leaf
        child_dict[iso] = deque([iso]) # The cells contained in the leaf are child?
        iso_list.append(iso)
        eic_list.append([])

    indices = iter(range(cutoff))

    min_active = 1
    timer()
    for i in indices:
        cell = cells_ordered[i] # loop through the cells, in the order of increasing potential
        ngb_labels = set(labels[
            pcn[cell] # labels of neighbors of this cell. Note that potential minima are
        ])              # already labeled with their flattened index.
        # a neighbor is previously explored but not isod (boundary), deactivate isos
        # TODO(SMOON) better name? what collide does?
        flag_deactivate = True if -2 in ngb_labels else False
        # ngb : lesser neighbors
        ngb_labels.discard(-1) # remove unprocessed cells (nor potential minima, nor XXX)
        ngb_labels.discard(-2) # what is -2?
        ngb_parents = list(set(
            [parent_dict[lbl] for lbl in ngb_labels]
        )) # now, ngb_parents is a list of parents of my lesser neighbor
        num_ngb_parents = len(ngb_parents) # number of all my neighbor's parents
        # Mark this cell as already explored.
        labels[cell] = -2 # I am now processed. TODO(SMOON) Why -2 instead of -1?
        if num_ngb_parents == 0:
            # No neighboring parent TODO(SMOON) when this can happen except for the very beginning?
            print("Cell i = {} have no lesser neighbors".format(i))
            if cell in active_isos: # active_isos is a set of flattened indices of "active" isos
                labels[cell] = cell # label this cell by its flattend index
                print("This cell {} is in fact one of active_isos. Label this cell by its flattend index".format(cell))
        elif flag_deactivate: # What is this?
            print("Neighbors of Cell i = {} has been previously explored, but not isod. deactivate isos")
            # a neighbor is previously explored but not isod (boundary), deactivate isos
            _collide(active_isos, ngb_parents)
            min_active = 0
            if len(active_isos) == min_active:
                next(islice(indices, cutoff-i-1, cutoff-i-1), None)
        elif num_ngb_parents == 1:
            # only 1 neighbor, inherit
            parent = ngb_parents[0] # this is a flattend index of the parenting cell of neighbor
            if parent in active_isos:
                labels[cell] = parent # label me as belong to this parent
                iso_dict[parent].append(cell) # Add me to parent's child list
        else:
            # There are 2 or more neighbors to deal with
            _merge(active_isos, ngb_parents, cell, iso_dict, child_dict,
                   parent_dict, iso_list, eic_list, labels)
            if verbose:
#                print(i,' of ',cutoff,' cells ',
#                      len(active_isos),' minima')
                print("Processing cell i = {} among {} cells: merge is triggered."\
                       " len(active_isos) = {}".format(i, cutoff, len(active_isos)))
            if len(active_isos) == min_active:
                # skip up to next iso or end
                next(islice(indices, cutoff-i-1, cutoff-i-1), None)

    dt = timer('loop finished for ' + str(cutoff) + ' items')
    if verbose:
        print(str(dt/i) + ' per cell')
    if verbose:
        print(str(dt/cutoff) + ' per total cell')
    return iso_dict, labels, iso_list, eic_list


def calc_leaf(iso_dict, iso_list, eic_list):
    leaf_dict = {}
    eic_dict = dict(zip(iso_list, eic_list))

    # fsd = find-split-dict, for each split list isos that it owns
    fsd = {}
    for iso in iso_list:
        if iso not in iso_dict:
            continue
        split = _find_split(iso, eic_dict)
        if split in fsd:
            fsd[split].append(iso)
        else:
            fsd[split] = [split]


    for split in fsd:
        # split is a leaf node
        if len(eic_dict[split]) == 0:
            leaf_dict[split] = []
            # but split also owns nodes above with only 1 child
            for subiso in fsd[split]:
                if subiso in iso_dict:
                    leaf_dict[split] += iso_dict[subiso]
    return leaf_dict


def _find_split(iso, eic_dict):
    # For a given iso and child data eic_dict, find the point where iso splits
    # eic_dict = dict(zip(iso_list,eic_list))
    eics = eic_dict[iso]
    le = len(eics)

    # If only 1 child, recurse
    if le == 1:
        return _find_split(eics[0], eic_dict)
    # 0 child leaf node, or multiple children, return self
    else:
        return iso


def _collide(active_isos, ngb_parents, *args):
    # TODO(SMOON) add docstring
    for parent in ngb_parents:
        if parent in active_isos:
            active_isos.discard(parent)


def _merge(active_isos, ngb_parents, cell, iso_dict, child_dict, parent_dict,
           iso_list, eic_list, labels):
    # merge removes deactivated ngb_parents
    # subsume removes deactivated ngb_parents
    # hence, inactive ngb_parents must have come from collide
    # first check for inactive ngb_parents which must have come from collide
    # if any inactive ngb_parents, collide
    # if all ngb_parents are active, first subsume smallest isos.
    if set(ngb_parents) <= active_isos:
        _tree_subsume(active_isos, ngb_parents, cell, iso_dict, child_dict,
                      parent_dict, iso_list, eic_list, labels)
    else:
        _collide(active_isos, ngb_parents)
        return 
    eic = list(set(ngb_parents) & active_isos)
    # eic = list(ngb_parents)
    # start new iso, assign it to be its own child and parent
    iso_dict[cell] = deque([cell])
    labels[cell] = cell
    child_dict[cell] = deque([cell])
    parent_dict[cell] = cell

    # new parent iso is active
    active_isos.add(cell)
    iso_list.append(cell)
    eic_list.append(eic)
    # exclusive immediate children
    for iso in eic:
        # new parent cell owns all children of all merging isos
        # cell's children is children
        child_dict[cell] += child_dict[iso]
        # children's parent is cell.
        for child in child_dict[iso]:
            parent_dict[child] = cell
            # deactivate iso
        active_isos.discard(iso)
    

def _tree_subsume(active_isos, ngb_parents, cell, iso_dict, child_dict,
                  parent_dict, iso_list, eic_list, labels):
    min_cells = 27
    eic_dict = dict(zip(iso_list, eic_list))
    subsume_set = set()
    ncells_list = [None] * len(ngb_parents)
    cell_list = [None] * len(ngb_parents)
    for i in range(len(ngb_parents)):
        parent = ngb_parents[i]
        ncells = len(iso_dict[parent])
        if ncells < min_cells:
            # too small, try making bigger TODO(SMOON) ????
            cell_list[i] = _recursive_members(iso_dict, eic_dict, parent)
            ncells = len(cell_list[i])
            if ncells < min_cells:
                # still too small 
                subsume_set.add(i)
        else: # big enough
            cell_list[i] = iso_dict[parent]
        ncells_list[i] = ncells
    largest_i = np.argmax(ncells_list)
    largest_parent = ngb_parents[largest_i]
    subsume_set.discard(largest_i)
    # add too small to largest
    for i in subsume_set:
        # add smaller iso cells to larger dict
        iso_dict[largest_parent] += cell_list[i]
        # relabel smaller iso cells to larger
        labels[cell_list[i]] = largest_parent
        # delete smaller iso
        active_isos.remove(ngb_parents[i])
        iso_dict.pop(ngb_parents[i])
        # note any children of smaller iso would be too small 
        # and be previously subsumed
    

def _recursive_members(iso_dict, eic_dict, iso):
    # get all cells of iso
    output = []
    output += iso_dict[iso]
    for child_iso in eic_dict[iso]:
        output += _recursive_members(iso_dict, eic_dict, child_iso)
    return output
