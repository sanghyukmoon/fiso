# Uses fiso_mem_sp to set X, Y shearing periodic (see function shear_bcn)
# X faces are sheared in the Y direction (bound_axis=0,shear_axis=1)
# Z direction clipped

import fiso.ext.fiso_mem_sp as sp
import fiso.ext.fiso_tree as tree
from fiso.tools.tree_bound import compute

def find_leaf(data,cell_shear):
    sp.cell_shear = int(cell_shear)
    # returns iso_dict, iso_labels
    return sp.find(data)

def find_full(data,cell_shear):
    sp.cell_shear = int(cell_shear)
    tree.setup = sp.setup
    # returns iso_dict, iso_labels, iso_list, eic_list
    return tree.find(data)

def find_split(iso,eic_dict):
    # find the point where objects split
    eics = eic_dict[iso]
    le = len(eics)
    if le == 0:
        return iso
    elif le == 1:
        return find_split(eics[0],eic_dict)
    else:
        return iso

def calc_leaf(iso_dict,iso_list,eic_list):
    # calculate leaf from full
    leaf_dict = {}
    eic_dict = dict(zip(iso_list,eic_list))

    # find split dict
    fsd = {}
    for iso in iso_list:
        if iso not in iso_dict:
            continue
        split = find_split(iso,eic_dict)
        if split in fsd:
            fsd[split].append(iso)
        else:
            fsd[split] = [split]
    for split in fsd:
        if len(eic_dict[split]) == 0:
            leaf_dict[split] = []
            for subiso in fsd[split]:
                if subiso in iso_dict:
                    leaf_dict[split] += iso_dict[subiso]
    return leaf_dict
