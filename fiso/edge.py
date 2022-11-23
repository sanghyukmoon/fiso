# code for identifying the edge cells of a core. 

import numpy as np
from .ext.fiso_periodic import shear_bcn, compute_displacement

def iso_dict_edge(iso_dict,bi,bpcn,displacements):
    """Find edge cells for all iso objects

    Arguments
    ---------
    iso_dict: dictionary containing fiso objects
    bi: boundary index
    bpcn: boundary precomputed neighbors
    displacements:

    Return
    ------
    out_dict: dictionary containing edge cells for each fiso objects
    """
    out_dict = {}
    for iso in iso_dict.keys():
        out_dict[iso] = iso_edge(iso_dict[iso],shape,bi,bpcn,displacements)
    return out_dict

def iso_edge(iso,bi,bpcn,displacements):
    """Do XXX

    Arguments
    ---------
    iso: a single fiso object
    bi: boundary index
    bpcn: boundary precomputed neighbors
    displacements:

    Return
    ------
    """
    members = np.array(iso)
    memlen = len(members)
    # nli is neighbor list by index
    nli = members[:,None] + displacements[None]
    # correct boundary by locating the nli index if existent of each bi 
    # (boundary index)
    # and inserting the associated entry from bpcn 
    # (boundary precomputed neighbors)
    mem_as = np.argsort(members)
    mem_as_bi = np.searchsorted(members[mem_as],bi)
    nli[mem_as[mem_as_bi[mem_as_bi < memlen]]] = bpcn[mem_as_bi < memlen]
    # if any neighbor isn't a member, this member is edge
    return members[np.any(np.isin(nli,members,invert=True),axis=1)]

def make_cd_edge(cells_dict,cell_shear,shape):
    # Initialize
    out_cd = {}
    # Compute bi, bpcn (boundary neighbors)
    corner_bool = True
    bi,bpcn = shear_bcn(shape,cell_shear)
    displacements = compute_displacement(shape,corner_bool)

    for iso in cells_dict.keys():
        out_cd[iso] = iso_edge(cells_dict[iso],bi,bpcn,displacements)
        # edge cells
    return out_cd
