# code for identifying the edge cells of a core. 

import numpy as n

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
    members = n.array(iso)
    memlen = len(members)
    # nli is neighbor list by index
    nli = members[:,None] + displacements[None]
    # correct boundary by locating the nli index if existent of each bi 
    # (boundary index)
    # and inserting the associated entry from bpcn 
    # (boundary precomputed neighbors)
    mem_as = n.argsort(members)
    mem_as_bi = n.searchsorted(members[mem_as],bi)
    nli[mem_as[mem_as_bi[mem_as_bi < memlen]]] = bpcn[mem_as_bi < memlen]
    # if any neighbor isn't a member, this member is edge
    return members[n.any(n.isin(nli,members,invert=True),axis=1)]
