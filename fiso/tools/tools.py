# Tools for working with iso_dicts


import numpy as n

def general_dict(array,iso_dict,function):
    # maps a function(array,iso_dict[key]) to all keys and return dict
    if len(array.shape) > 1:
        array = array.reshape(-1)
    val_dict = {}
    for key in iso_dict.keys():
        val_dict[key] = function(array,iso_dict[key])
    return val_dict

def map_dict(array,iso_dict,function):
    # maps a function(array,iso_dict[key]) to all keys and return dict
    if len(array.shape) > 1:
        array = array.reshape(-1)
    val_dict = {}
    for key in iso_dict.keys():
        val_dict[key] = function(array[iso_dict[key]])
    return val_dict

def value_dict(array,iso_dict):
    # returns dict with array values
    if len(array.shape) > 1:
        array = array.reshape(-1)
    val_dict = {}
    for key in iso_dict.keys():
        val_dict[key] = array[iso_dict[key]]
    return val_dict

def sum_dict(array,iso_dict):
    return map_dict(array,iso_dict,n.sum)

def len_dict(array,iso_dict):
    return map_dict(array,iso_dict,len)

def sort_to_list(val_dict):
    sorted_keys = n.sort(val_dict.keys())
    lsk = len(sorted_keys)
    out_list = [None]*lsk
    for i in range(lsk):
        out_list[i] = val_dict[sorted_keys[i]]
    return out_list

def sum_list(array,iso_dict):
    return sort_to_list(sum_dict(array,iso_dict))

def len_list(array,iso_dict):
    return sort_to_list(len_dict(array,iso_dict))

def filter_dict(array,iso_dict):
    # keep only values that are in iso_dict
    index = []
    for value in iso_dict.values():
        index += list(value)
    flat = array.reshape(-1)
    output = n.full(len(flat), n.nan)
    output[index] = 1.0*flat[index]
    return output.reshape(array.shape)

def find_split(iso,eic_dict):
    # For a given iso and child data eic_dict, find the point where iso splits
    # eic_dict = dict(zip(iso_list,eic_list))
    eics = eic_dict[iso]
    le = len(eics)
    
    # If only 1 child, recurse
    if le == 1:
        return find_split(eics[0],eic_dict)
    # 0 child leaf node, or multiple children, return self
    else:
        return iso

def calc_leaf(iso_dict,iso_list,eic_list):
    leaf_dict = {}
    eic_dict = dict(zip(iso_list,eic_list))

    # fsd = find-split-dict, for each split list isos that it owns
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
        # split is a leaf node
        if len(eic_dict[split]) == 0:
            leaf_dict[split] = []
            # but split also owns nodes above with only 1 child 
            for subiso in fsd[split]:
                if subiso in iso_dict:
                    leaf_dict[split] += iso_dict[subiso]
    return leaf_dict
