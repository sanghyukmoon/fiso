# Tools for working with iso_dicts
import numpy as np
import xarray as xr

def filter_dict(array,iso_dict):
    # keep only values that are in iso_dict
    index = []
    for value in iso_dict.values():
        index += list(value)
    flat = array.reshape(-1)
    output = np.full(len(flat), np.nan)
    output[index] = 1.0*flat[index]
    return output.reshape(array.shape)

def filter_var(var, iso_dict=None, iso=None):
    """Set var = 0 outside the region defined by iso_dict

    Arguments
    ---------
    var: xarray.DataArray
        variable to be filtered (rho, phi, etc.)
    iso_dict: FISO object dictionary
    iso (optional): int
        id (dict key) of the object to select

    Return
    ------
    res: Filtered DataArray
    """
    if iso_dict==None:
        return var
    if iso:
        iso_dict = dict(iso=iso_dict[iso])
    res = filter_dict(var.data, iso_dict)
    res = xr.DataArray(data=res, coords=var.coords, dims=var.dims)
    return res

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
