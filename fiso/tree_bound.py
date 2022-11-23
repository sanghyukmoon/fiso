# Compute bound mass in a hierarchical way
import numpy as np
from .tools import find_split

def compute(data,iso_dict,iso_list,eic_list):
    # New bound mass must contain self, otherwise don't keep climbing up

    # Initialize
    len_iso = len(iso_list)
    cells_dict = {}
    bmass_dict = {}
    bcell_dict = {}
    eic_dict = dict(zip(iso_list,eic_list))

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
        cells_dict[iso] = []
        cells_dict[iso] += iso_dict[iso]
        for eic in eic_dict[iso]:
            cells_dict[iso] += cells_dict[eic]

        # 2. Calculate Bound Mass

        bcell_dict[iso],bmass_dict[iso] = bound_mass(data,cells_dict[iso],0.0)

        # 3. Merge Logic
        split_iso = find_split(iso,eic_dict)
        if split_iso in bcell_dict[iso]:
            # this iso can replace its children
            # because its starting point was bound and it joins the children
            # remove children from output cell dict
            for eic in eic_dict[iso]:
                if eic in cells_dict.keys():
                    cells_dict.pop(eic)
                if eic in bcell_dict.keys():
                    bcell_dict.pop(eic)
            # set this iso to be a leaf node
            eic_dict[iso] = []
        else:
            # this iso cannot replace its children
            if len(eic_dict[iso]) > 0:
            # this iso has children so it is removed
                if iso in cells_dict.keys():
                    cells_dict.pop(iso)
                if iso in bcell_dict.keys():
                    bcell_dict.pop(iso)
    return cells_dict,bcell_dict

def bound_mass(data,cells,e0):
    # assume data is the following:
    cells = np.array(cells)
    rho,phi,pressure,bpressure,velx,vely,velz = data
    pre_phi = phi.flatten()[cells]
    order = np.argsort(pre_phi)
    c_phi = pre_phi[order]
    c_rho = rho.flatten()[cells][order]
    c_p = pressure.flatten()[cells][order]
    c_b = bpressure.flatten()[cells][order]
    c_x = velx.flatten()[cells][order]
    c_y = vely.flatten()[cells][order]
    c_z = velz.flatten()[cells][order]

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
    threshold = np.where(c_tot < e0*np.arange(1,1+len(c_tot)) )[0]
    if len(threshold) < 1:
        return cells[order][:0], 0.0
    else:
        index = threshold[-1]
        return cells[order][:index], cc_rho[index]

def recursive_members(iso_dict,eic_dict,iso):
    # get all cells of iso
    output = []
    output += iso_dict[iso]
    for child_iso in eic_dict[iso]:
        output += recursive_members(iso_dict,eic_dict,child_iso)
    return output

def recursive_child(iso_dict,eic_dict,iso):
    # get all eic descendents of iso
    output = []
    # print(eic_dict[iso])
    output += eic_dict[iso]
    for child_iso in eic_dict[iso]:
        output += recursive_child(iso_dict,eic_dict,child_iso)
    return output
