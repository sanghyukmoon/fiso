import numpy as np
import xarray as xr

def filter_var(var, iso_dict=None, cells=None):
    """Set var = 0 outside the region defined by iso_dict

    Parameters
    ----------
    var : xarray.DataArray
        Variable to be filtered (rho, phi, etc.)
    iso_dict : dict, optional
        FISO object dictionary
    cells : list or array-like, optional
        Flattend indices

    Return
    ------
    out : Filtered DataArray
    """
    if iso_dict is None and cells is None:
        return var
    elif iso_dict is not None and cells is not None:
        raise ValueError("Either give iso_dict or cells, but not both")
    elif cells is None:
        cells = []
        for value in iso_dict.values():
            cells += value
    var_flat = var.data.flatten()
    out = np.full(len(var_flat), np.nan)
    out[cells] = var_flat[cells]
    out = out.reshape(var.shape)
    out = xr.DataArray(data=out, coords=var.coords, dims=var.dims)
    return out

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
