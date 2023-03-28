# This extension returns extra objects
# iso_list: for keeping track of all members of the tree
# eic_list: for keeping track of the tree structure, the immediate children
# of each member

from collections import deque
from itertools import islice
import numpy as np
from scipy.ndimage import minimum_filter
from .tools import timer
from .edge import precompute_neighbor


def construct_tree(data, boundary_flag, verbose=True):
    """Construct isocontour tree

    Parameters
    ----------
    data : numpy.ndarray
        Input array
    boundary_flag: str (periodic | outflow)
        Determines how the input array is extended when the
        stencil for finding the local minima overlaps a border.

    Returns
    -------
    iso_dict : dict
        Dictionary containing all isos.
    labels : list
    iso_list : list
    eic_list : list
    """

    timer()
    # Prepare data
    cells_ordered = data.flatten().argsort() # sort phi in ascending order.
    Ncells = len(cells_ordered)
    timer('sort')

    # Precompute neighbor indices
    pcn = precompute_neighbor(data.shape, boundary_flag, corner=True)
    timer('precompute neighbor indices')

    # Find local minima
    # minima_flat is flattened boolian mask
    minima_flat = find_minima_global(data, boundary_flag).flatten()
    # 1D flattened index array of potential minima
    idx_minima = np.where(minima_flat)[0]
    if verbose:
        print("Found {} minima".format(len(idx_minima)))


    # iso dict and labels setup
    iso_dict = {}
    labels = -np.ones(Ncells, dtype=int) # indices are real index locations
    # inside loop, labels are accessed by labels[cells_ordered[i]]
    for mini in idx_minima:
        iso_dict[mini] = deque([mini])
    labels[idx_minima] = idx_minima
    active_isos = set(idx_minima)

    iso_list = []
    eic_list = [] # exclusive immediate children
    parent_dict = {}
    child_dict = {}
    for iso in active_isos:
        # The potential minimum "parents" the cells contained in the leaf
        parent_dict[iso] = iso
        # The cells contained in the leaf are childs of this iso
        child_dict[iso] = deque([iso])
        iso_list.append(iso)
        eic_list.append([])


    min_active = 1
    timer()
    indices = iter(range(Ncells))
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
            if cell in active_isos: # active_isos is a set of flattened indices of "active" isos
                labels[cell] = cell # label this cell by its flattend index
        elif flag_deactivate: # What is this?
            # a neighbor is previously explored but not isod (boundary), deactivate isos
            _collide(active_isos, ngb_parents)
            min_active = 0
            if len(active_isos) == min_active:
                next(islice(indices, Ncells-i-1, Ncells-i-1), None)
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
                print("i = {} of {} cells: Reaching critical point."\
                       " len(active_isos) = {}".format(i, Ncells, len(active_isos)))
            if len(active_isos) == min_active:
                # skip up to next iso or end
                next(islice(indices, Ncells-i-1, Ncells-i-1), None)

    dt = timer('loop finished for {} items'.format(Ncells))
    if verbose:
        print('{} per cell'.format(dt/i))
    if verbose:
        print('{} per total cell'.format(dt/Ncells))
    return iso_dict, labels, iso_list, eic_list


def calc_leaf(iso_dict, iso_list, eic_list):
    """Calculate leaf HBPs

    Parameters
    ----------
    iso_dict : dict
        iso_dict returned by construct_tree
    iso_list : list
        iso_list returned by construct_tree
    eic_list : list
        eic_list returned by construct_tree

    Returns
    -------
    leaf_dict : dict
        Dictionary containing all leaf HBPs.
    """
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
        if len(eic_dict[split]) == 0:
            leaf_dict[split] = []
            # but split also owns nodes above with only 1 child
            for subiso in fsd[split]:
                if subiso in iso_dict:
                    leaf_dict[split] += iso_dict[subiso]
    return leaf_dict


def find_minima_global(arr, boundary_flag):
    """Find local minima of the input array

    Parameters
    ----------
    arr : array_like
        Input array
    boundary_flag: str
        boundary flag determines how the input array is extended when the
        stencil for finding the local minima overlaps a border.

    Returns
    -------
    local_min : array_like
        Bolean array that selects local minima
    """
    if boundary_flag == 'periodic':
        mode = 'wrap'
    elif boundary_flag == 'outflow':
        mode = 'reflect'
    else:
        raise Exception("unknown boundary mode")
    local_min = (arr == minimum_filter(arr, size=3, mode=mode))
    return local_min

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
    # loop through all parents except for the largest one
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
