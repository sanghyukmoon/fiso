# Compute bound mass in a hierarchical way
import numpy as np
from .fiso_tree import _find_split


def compute(data, iso_dict, iso_list, eic_list):
    # New bound mass must contain self, otherwise don't keep climbing up

    # Initialize
    len_iso = len(iso_list)
    hpr_dict = {}
    hbr_dict = {}
    eic_dict = dict(zip(iso_list, eic_list))

    # Loop over isos
    for i in range(len_iso):
        # 1. Get all member cells of iso
        iso = iso_list[i]
        if iso not in iso_dict.keys():
            # must be in iso_dict
            continue

        if any([(len(eic_dict[eic]) > 0) for eic in eic_dict[iso]]):
            # immediate children must be leaf nodes
            continue

        # for all leaf nodes
        # get the cells of self and inherit from immediate children
        # TODO(SMOON) there is some inconsistency between these lines and the
        # similar part in calc_leaf, which might have caused the leaf-HPR bug.
        hpr_dict[iso] = []
        hpr_dict[iso] += iso_dict[iso]
        for eic in eic_dict[iso]:
            hpr_dict[iso] += hpr_dict[eic]

        # 2. Calculate Bound Mass

        hbr_dict[iso] = bound_region(data, hpr_dict[iso], 0.0)

        # 3. Merge Logic
        split_iso = _find_split(iso, eic_dict)
        if split_iso in hbr_dict[iso]:
            # this iso can replace its children
            # because its starting point was bound and it joins the children
            # remove children from output cell dict
            for eic in eic_dict[iso]:
                if eic in hpr_dict.keys():
                    hpr_dict.pop(eic)
                if eic in hbr_dict.keys():
                    hbr_dict.pop(eic)
            # set this iso to be a leaf node
            eic_dict[iso] = []
        else:
            # this iso cannot replace its children
            if len(eic_dict[iso]) > 0:
                # this iso has children so it is removed
                if iso in hpr_dict.keys():
                    hpr_dict.pop(iso)
                if iso in hbr_dict.keys():
                    hbr_dict.pop(iso)
    return hpr_dict, hbr_dict


def bound_region(data, cells, e0):
    # assume data is the following:
    cells = np.array(cells)
    rho, phi, pressure, bpressure, velx, vely, velz = data
    pre_phi = phi.flatten()[cells]
    cells_ordered = np.argsort(pre_phi)
    c_phi = pre_phi[cells_ordered]
    c_rho = rho.flatten()[cells][cells_ordered]
    c_p = pressure.flatten()[cells][cells_ordered]
    c_b = bpressure.flatten()[cells][cells_ordered]
    c_x = velx.flatten()[cells][cells_ordered]
    c_y = vely.flatten()[cells][cells_ordered]
    c_z = velz.flatten()[cells][cells_ordered]

    c_phi0 = c_phi[-1]

    # compute center of mass quantities
    cc_rho = np.cumsum(c_rho)
    cc_x0 = np.cumsum(c_rho*c_x)
    cc_y0 = np.cumsum(c_rho*c_y)
    cc_z0 = np.cumsum(c_rho*c_z)

    c_com = 0.5 * (cc_x0*cc_x0 + cc_y0*cc_y0 + cc_z0*cc_z0)/cc_rho

    c_tot = np.cumsum((c_phi - c_phi0) * c_rho +
                      + 1.5*c_p
                      + c_b
                      + 0.5 * c_rho * (c_x*c_x + c_y*c_y + c_z*c_z)
                      ) - c_com
    threshold = np.where(c_tot < e0*np.arange(1, 1+len(c_tot)))[0]
    if len(threshold) < 1:
        return cells[cells_ordered][:0]
    else:
        index = threshold[-1]
        return cells[cells_ordered][:index]
