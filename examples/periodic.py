# Uses fiso_mem_sp to set X, Y shearing periodic (see function shear_bcn)
# X faces are sheared in the Y direction (bound_axis=0,shear_axis=1)
# Z direction clipped

import fiso.ext.fiso_tree as tree
tree.fiso.boundary_mode = 'wrap' # makes boundaries periodic
import fiso.tools.tree_bound as bound
import fiso.tools.contour

def find(data):
    return tree.find(data)
    # returns iso_dict, iso_labels, iso_list, eic_list
    return tree.find(data)

def compute_bound(data,iso_dict,iso_list,eic_list):
    # data here is rho phi pressure bpressure velx vely velz
    return bound.compute(data,iso_dict,iso_list,eic_list)

def 

find_split = bound.find_split

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
