import numpy as n

def general_dict(array,iso_dict,function):
    if len(array.shape) > 1:
        array = array.reshape(-1)
    val_dict = {}
    for key in iso_dict.keys():
        val_dict[key] = function(array,iso_dict[key])
    return val_dict

def map_dict(array,iso_dict,function):
    if len(array.shape) > 1:
        array = array.reshape(-1)
    val_dict = {}
    for key in iso_dict.keys():
        val_dict[key] = function(array[iso_dict[key]])
    return val_dict

def value_dict(array,iso_dict):
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

