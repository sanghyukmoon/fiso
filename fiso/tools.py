import numpy as np
import xarray as xr
from .edge import get_edge_cells

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
            cells += list(value)
    var_flat = var.data.flatten()
    out = np.full(len(var_flat), np.nan)
    out[cells] = var_flat[cells]
    out = out.reshape(var.shape)
    out = xr.DataArray(data=out, coords=var.coords, dims=var.dims)
    return out

def get_center(iso_id, phi):
    """return coordinates at the potential minimum
    Parameters
    ----------
    iso_id : int
    dat : xarray.Dataset
    """
    center = phi.argmin(...)
    x0, y0, z0 = [phi.isel(center).coords[dim].data[()] for dim in ['x','y','z']]
    return x0, y0, z0

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

def get_energies(dat, cells, mode, pcn=None):
    """Calculate cumulative energies for all levels

    Arguments
    ---------
    dat : xr.Dataset
        input dataset containing density, velocity, pressure, and potential
    cells : list or array-like, optional
        Flattend indices
    mode : str
        Definition of the boundness. Options are:
        'HBR', 'HBR+1', 'HBR-1', or 'virial'
    pcn : array_like
        precomputed neighbors.

    Return
    ------
    energies : dict
        Integrated energies and effective radius at each level
    """
    # Get hydro variables for a selected HPR
    cells = np.array(cells)
    dat = dat.transpose('z','y','x')
    dat1d = dict(cells = cells,
                 x = np.broadcast_to(dat.x.data,dat.dens.shape
                     ).flatten()[cells],
                 y = np.broadcast_to(dat.y.data, dat.dens.shape
                     ).transpose(1,2,0).flatten()[cells],
                 z = np.broadcast_to(dat.z.data, dat.dens.shape
                     ).transpose(2,0,1).flatten()[cells],
                 rho = dat.dens.data.flatten()[cells],
                 vx = dat.vel1.data.flatten()[cells],
                 vy = dat.vel2.data.flatten()[cells],
                 vz = dat.vel3.data.flatten()[cells],
                 prs = dat.prs.data.flatten()[cells],
                 phi = dat.phi.data.flatten()[cells])
    # Assume uniform grid
    dx = (dat.x[1]-dat.x[0]).data[()]
    dy = (dat.y[1]-dat.y[0]).data[()]
    dz = (dat.z[1]-dat.z[0]).data[()]
    dV = dx*dy*dz
    gm1 = (5./3. - 1)

    # Sort variables in ascending order of potential
    order = dat1d['phi'].argsort()
    dat1d = {key:value[order] for key, value in dat1d.items()}
    cells = dat1d['cells']

    # Gravitational potential at the HPR boundary
    phi0 = dat1d['phi'][-1]

    # Calculate the center of momentum frame
    # note: dV factor is omitted
    M = (dat1d['rho']).cumsum()
    vx0 = (dat1d['rho']*dat1d['vx']).cumsum() / M
    vy0 = (dat1d['rho']*dat1d['vy']).cumsum() / M
    vz0 = (dat1d['rho']*dat1d['vz']).cumsum() / M
    # Potential minimum
    phi_hpr = filter_var(dat.phi, cells=cells)
    k0, j0, i0 = map(lambda x: x.data[()], phi_hpr.argmin(...).values())
    z0 = dat.z.isel(z=k0).data[()]
    y0 = dat.y.isel(y=j0).data[()]
    x0 = dat.x.isel(x=i0).data[()]

    # Kinetic energy
    # \int 0.5 \rho | v - v_com |^2 dV
    # = \int 0.5 \rho |v|^2 dV - (\int 0.5\rho dV) |v_com|^2
    # Note that v_com depends on the limit of the volume integral.
    Ekin = (0.5*dat1d['rho']*(dat1d['vx']**2
                              + dat1d['vy']**2
                              + dat1d['vz']**2)*dV).cumsum()
    Ekin -= (0.5*dat1d['rho']*dV).cumsum()*(vx0**2 + vy0**2 + vz0**2)

    # Thermal energy
    Eth = (dat1d['prs']/gm1*dV).cumsum()

    # Gravitational energy
    if mode=='HBR' or mode=='HBR+1' or mode=='HBR-1':
        Egrav = (dat1d['rho']*(dat1d['phi'] - phi0)*dV).cumsum()
    elif mode=='virial':
        dat1d['gx'] = -dat.phi.differentiate('x').data.flatten()[cells]
        dat1d['gy'] = -dat.phi.differentiate('y').data.flatten()[cells]
        dat1d['gz'] = -dat.phi.differentiate('z').data.flatten()[cells]
        Egrav = (dat1d['rho']*((dat1d['x'] - x0)*dat1d['gx']
                               + (dat1d['y'] - y0)*dat1d['gy']
                               + (dat1d['z'] - z0)*dat1d['gz'])*dV).cumsum()
    else:
        raise ValueError("Unknown mode; select (HBR | HBR+1 | HBR-1 | virial)")

    # Surface terms
    if mode=='HBR':
        Ekin0 = Eth0 = np.zeros(len(cells))
    elif mode=='HBR+1' or mode=='HBR-1':
        if pcn is None:
            raise Exception("HBR+-1 calculation requires pcn")
        edge = get_edge_cells(cells, pcn)
        edg1d = dict(rho = dat.dens.data.flatten()[edge],
                     vx = dat.vel1.data.flatten()[edge],
                     vy = dat.vel2.data.flatten()[edge],
                     vz = dat.vel3.data.flatten()[edge],
                     prs = dat.prs.data.flatten()[edge])
        # COM velocity of edge cells
        M = (edg1d['rho']).sum()
        vx0 = (edg1d['rho']*edg1d['vx']).sum() / M
        vy0 = (edg1d['rho']*edg1d['vy']).sum() / M
        vz0 = (edg1d['rho']*edg1d['vz']).sum() / M
        # Mean surface energies
        Ekin0 = (0.5*edg1d['rho']*((edg1d['vx'] - vx0)**2
                                   + (edg1d['vy'] - vy0)**2
                                   + (edg1d['vz'] - vz0)**2)).mean()
        Eth0 = (edg1d['prs']/gm1).mean()
        # Integrated surface energy to compare with volume energies.
        # Note that the excess energy is given by \int (E - E_0) dV
        Ekin0 = (Ekin0*np.ones(len(cells))*dV).cumsum()
        Eth0 = (Eth0*np.ones(len(cells))*dV).cumsum()
    elif mode=='virial':
        divPx = ((dat.prs*(dat.x - x0)).differentiate('x')
                 + (dat.prs*(dat.y - y0)).differentiate('y')
                 + (dat.prs*(dat.z - z0)).differentiate('z')
                 ).data.flatten()[cells]
        Eth0 = ((1./3.)*divPx/gm1*dV).cumsum()
        # Kinetic energy surface term
        v0 = np.array([vx0, vy0, vz0])
        # A1
        rho_rdotv = dat.dens*((dat.x - x0)*dat.vel1
                             + (dat.y - y0)*dat.vel2
                             + (dat.z - z0)*dat.vel3)
        A1 = ((rho_rdotv*dat.vel1).differentiate('x')
              + (rho_rdotv*dat.vel2).differentiate('y')
              + (rho_rdotv*dat.vel3).differentiate('z'))
        A1 = (A1.data.flatten()[cells]*dV).cumsum()
        # A2
        grad_rho_r = np.empty((3,3), dtype=xr.DataArray)
        for i, crd_i in enumerate(['x','y','z']):
            for j, (crd_j, pos0_j) in enumerate(zip(['x','y','z'], [x0,y0,z0])):
                grad_rho_r[i,j] = (dat.dens*(dat[crd_j] - pos0_j)
                                   ).differentiate(crd_i)
        A2 = np.empty((3,3,len(cells)))
        for i, crd_i in enumerate(['x','y','z']):
            for j, (crd_j, pos0_j) in enumerate(zip(['x','y','z'], [x0,y0,z0])):
                A2[i,j,:] = (grad_rho_r[i,j].data.flatten()[cells]*dV).cumsum()
        A2 = np.einsum('i..., ij..., j...', v0, A2, v0)
        # A3
        grad_rho_rdotv = np.empty(3, dtype=xr.DataArray)
        for i, crd_i in enumerate(['x','y','z']):
            grad_rho_rdotv[i] = rho_rdotv.differentiate(crd_i)
        A3 = np.empty((3,len(cells)))
        for i, crd_i in enumerate(['x','y','z']):
            A3[i,:] = (grad_rho_rdotv[i].data.flatten()[cells]*dV).cumsum()
        A3 = np.einsum('i...,i...', v0, A3)
        # A4
        div_rhorv = np.empty(3, dtype=xr.DataArray)
        for i, (crd_i, pos0_i) in enumerate(zip(['x','y','z'], [x0,y0,z0])):
            div_rhorv[i] = ((dat.dens*(dat[crd_i] - pos0_i)*dat.vel1).differentiate('x')
                            + (dat.dens*(dat[crd_i] - pos0_i)*dat.vel2).differentiate('y')
                            + (dat.dens*(dat[crd_i] - pos0_i)*dat.vel3).differentiate('z'))
        A4 = np.empty((3,len(cells)))
        for i, crd_i in enumerate(['x','y','z']):
            A4[i,:] = (div_rhorv[i].data.flatten()[cells]*dV).cumsum()
        A4 = np.einsum('i...,i...', v0, A4)
        Ekin0 = 0.5*(A1 + A2 - A3 - A4)

    Reff = ((np.ones(len(cells))*dV).cumsum() / (4.*np.pi/3.))**(1./3.)
    if mode=='HBR':
        Etot = Ekin + Eth + Egrav
    elif mode=='HBR+1':
        Etot = (Ekin - Ekin0) + (Eth - Eth0) + Egrav
    elif mode=='HBR-1':
        Etot = (Ekin + Ekin0) + (Eth + Eth0) + Egrav
    elif mode=='virial':
        Etot = 2*(Ekin - Ekin0) + 3*gm1*(Eth - Eth0) + Egrav

    energies = dict(Reff=Reff, Ekin=Ekin, Eth=Eth, Ekin0=Ekin0, Eth0=Eth0,
                    Egrav=Egrav, Etot=Etot)
    return energies

def get_Etot(dat, cells, level, mode, pcn=None, return_all=False, cumulative=True):
    """Calculate total energy below a given isocontour level for a given HPR

    Arguments
    ---------
    dat: TODO
    level: int
        isocontour level (integer index for phi_sorted)
    mode: "bound" or "virial"
        definition of the "core"
    full: boolean
        if True, return Eth, Ekin, Ekin0, Egrav

    Return
    ------
    Etot: float
        Total energy-like contained in the isocontour level
    """
    # Get hydro variables for a selected HPR
    hpr_cells = np.array(cells)
    dat = dat.transpose('z','y','x')
    dat1d = dict(
# x = np.broadcast_to(dat.x.data, dat.dens.shape).flatten()[hpr_cells],
# y = np.broadcast_to(dat.y.data, dat.dens.shape).transpose(1,2,0).flatten()[hpr_cells],
# z = np.broadcast_to(dat.z.data, dat.dens.shape).transpose(2,0,1).flatten()[hpr_cells],
                 rho = dat.dens.data.flatten()[hpr_cells],
                 vx = dat.vel1.data.flatten()[hpr_cells],
                 vy = dat.vel2.data.flatten()[hpr_cells],
                 vz = dat.vel3.data.flatten()[hpr_cells],
                 prs = dat.prs.data.flatten()[hpr_cells],
                 phi = dat.phi.data.flatten()[hpr_cells])
    # Assume uniform grid
    dx = (dat.x[1]-dat.x[0]).data[()]
    dy = (dat.y[1]-dat.y[0]).data[()]
    dz = (dat.z[1]-dat.z[0]).data[()]
    dV = dx*dy*dz
    gm1 = (5./3. - 1)

    # Sort variables in ascending order of potential
    order = dat1d['phi'].argsort()
    dat1d = {key:value[order] for key, value in dat1d.items()}

    # Gravitational potential at the HPR boundary
    phi0 = dat1d['phi'][-1]

    # Select cells below a given level (inclusive)
    dat1d = {key:value[:level+1] for key, value in dat1d.items()}
    Vol = level*dV # Volume of the selected region
    cells_srtd = hpr_cells[order[:level+1]]

    # Calculate the center of momentum frame
    M = dat1d['rho'].sum()
    vx0 = (dat1d['rho']*dat1d['vx']).sum() / M
    vy0 = (dat1d['rho']*dat1d['vy']).sum() / M
    vz0 = (dat1d['rho']*dat1d['vz']).sum() / M
    # Center of mass position
#     x0 = (dat1d['rho']*dat1d['x']).sum() / M
#     y0 = (dat1d['rho']*dat1d['y']).sum() / M
#     z0 = (dat1d['rho']*dat1d['z']).sum() / M
    # Potential minimum
    phi_hpr = filter_var(dat.phi, cells=hpr_cells)
    k0, j0, i0 = map(lambda x: x.data[()], phi_hpr.argmin(...).values())
    z0 = dat.z.isel(z=k0).data[()]
    y0 = dat.y.isel(y=j0).data[()]
    x0 = dat.x.isel(x=i0).data[()]

    if not cumulative:
        # Select a cell at a given level
        dat1d = {key:value[level] for key, value in dat1d.items()}
        cells_srtd = hpr_cells[order[level]]
        dV = Vol = 1

    # Kinetic energy
    Ekin = 0.5*(dat1d['rho']*((dat1d['vx']-vx0)**2
                              + (dat1d['vy']-vy0)**2
                              + (dat1d['vz']-vz0)**2)*dV).sum()
    # Thermal energy
    Eth = (dat1d['prs']/gm1*dV).sum()

    # Gravitational energy
    if mode=='HBR' or mode=='HBR+1' or mode=='HBR-1':
        Egrav = (dat1d['rho']*(dat1d['phi'] - phi0)*dV).sum()
    elif mode=='virial':
        gx = -dat.phi.differentiate('x')
        gy = -dat.phi.differentiate('y')
        gz = -dat.phi.differentiate('z')
        Egrav = (dat.dens*((dat.x-x0)*gx
                           + (dat.y-y0)*gy
                           + (dat.z-z0)*gz)*dV).data.flatten()[cells_srtd].sum()
    else:
        raise ValueError("Unknown mode; Select mode = (HBR | HBR+1 | HBR-1 | virial)")

    # Surface terms
    if mode=='HBR':
        Ekin0 = Eth0 = 0
    if mode=='HBR+1' or mode=='HBR-1':
        if pcn is None:
            raise Exception("HBR+-1 calculation requires pcn")
        edge_cells = get_edge_cells(hpr_cells, pcn)
        Ekin0 = (0.5*dat.dens*((dat.vel1 - vx0)**2
                               + (dat.vel2 - vy0)**2
                               + (dat.vel3 - vz0)**2
                               )).data.flatten()[edge_cells].mean()*Vol
        Eth0 = (dat.prs/gm1).data.flatten()[edge_cells].mean()*Vol
    elif mode=='virial':
        rho_rdotv = dat.dens*((dat.x - x0)*(dat.vel1 - vx0)
                              + (dat.y - y0)*(dat.vel2 - vy0)
                              + (dat.z - z0)*(dat.vel3 - vz0))
        Ekin0 = 0.5*(((rho_rdotv*(dat.vel1 - vx0)).differentiate('x')
                      + (rho_rdotv*(dat.vel2 - vy0)).differentiate('y')
                      + (rho_rdotv*(dat.vel3 - vz0)).differentiate('z')
                     )*dV).data.flatten()[cells_srtd].sum()
        Eth0 = (1.0/(3*gm1)*((dat.prs*(dat.x - x0)).differentiate('x')
                             + (dat.prs*(dat.y - y0)).differentiate('y')
                             + (dat.prs*(dat.z - z0)).differentiate('z')
                            )*dV).data.flatten()[cells_srtd].sum()

    if mode=='HBR':
        Etot = Ekin + Eth + Egrav
    elif mode=='HBR+1':
        Etot = (Ekin - Ekin0) + (Eth - Eth0) + Egrav
    elif mode=='HBR-1':
        Etot = (Ekin + Ekin0) + (Eth + Eth0) + Egrav
    elif mode=='virial':
        Etot = 2*(Ekin - Ekin0) + 3*gm1*(Eth - Eth0) + Egrav

    if return_all:
        return dict(Ekin=Ekin, Eth=Eth, Ekin0=Ekin0, Eth0=Eth0, Egrav=Egrav, Etot=Etot)
    else:
        return Etot

def groupby_bins(dat, coord, edges):
    """Alternative to xr.groupby_bins, which is very slow

    Arguments
    ---------
    dat: xarray.DataArray
        input dataArray
    coord: str
        coordinate name
    edges: array-like
        bin edges

    Return
    ------
    res: xarray.DataArray
        binned array
    """
    dat = dat.transpose('z','y','x')
    fc = dat[coord].data.flatten() # flattened coordinates
    fd = dat.data.flatten() # flattened data
    res = np.histogram(fc, edges, weights=fd)[0] / np.histogram(fc, edges)[0]
    rc = 0.5*(edges[1:] + edges[:-1])
    res = xr.DataArray(data=res, coords=dict(r=rc), name=dat.name)
    return res
