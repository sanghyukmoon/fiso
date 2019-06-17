# a few tools to help plot isocontours

import sys
import numpy as n
import matplotlib.colors as mco
import pylab as p
import scipy.ndimage as sn
import fiso.tools.tools as ftt

def mesh(x,y,z):
    p.pcolormesh(x,y,z,norm=mco.LogNorm(),shading='flat',edgecolors='none')

def project(field):
    return n.max(field,axis=2).transpose()

def surface(field):
    return n.sum(field,axis=2).transpose()

def find_minima(arr,mode):
    arrmin = n.min(arr)
    smaller = -2.0*n.abs(arrmin)                                           
    nhbd = sn.generate_binary_structure(len(arr.shape),mode) #neighborhood      
    sfmf = (sn.filters.minimum_filter(arr, footprint=nhbd)) 
    local_min = (sfmf == arr)
    min_coords = n.where(local_min)
    posx = min_coords[0]
    posy = min_coords[1]
    return posx,posy     

def calc_slice(core_dict,index,shape):
    cindex = core_dict[index]
    coords = n.unravel_index(cindex,shape)
    center_coord = n.unravel_index(index,shape)[2] #z coordinate of minimum
    newcoords = [None]*len(coords)
    slicecoords = [None]*len(coords)
    cmax = n.max(coords,axis=1) #max of each dimension
    cmin = n.min(coords,axis=1) #min of each dimension
    ccenter = (cmax+cmin)/2 #center of each dimension
    csize = n.max(cmax - cmin) #size of each dimension
    slices = len(cmax)*[None]
    for i in range(len(cmax)):
        smin = max(ccenter[i]-csize,0)
        smax = min(ccenter[i]+csize,shape[i])
        slices[i] = slice(int(smin),int(smax)) 
        # how to slice 3D array
        newcoords[i] = (coords[i] - smin ).astype(int)
        # coordinates for 3-D projected mask
        slicecoords[i] = newcoords[i][coords[2] == center_coord] 
        # coords for 2-D planar mask
    return slices, newcoords, slicecoords

def calc_plot(rho,phi,core_dict,index):
    slices, newcoords, slicecoords = calc_slice(core_dict,index,rho.shape)
    center_coord = n.unravel_index(index,rho.shape)[2]
    core_phi = phi[tuple(slices)][tuple(newcoords)]
    phi0 = n.min(core_phi)
    phi1 = n.max(core_phi)
    conres = 5.
    dphi = (phi1-phi0)/conres
    phi_levels = n.arange(phi0,phi1+conres*dphi,dphi)
    slices[2] = center_coord
    x = n.arange(slices[0].start,slices[0].stop)
    y = n.arange(slices[1].start,slices[1].stop)
    srho = rho[tuple(slices)]
    sphi = phi[tuple(slices)]
    mask = n.zeros(srho.shape)
    mask[tuple(newcoords[:2])] = 1.0
    slicemask = n.zeros(srho.shape)
    slicemask[tuple(slicecoords[:2])] = 1.0
    return x,y,phi_levels,srho.transpose(),sphi.transpose(),mask.transpose(),slicemask.transpose()

def plot_fiso_slice(rho,phi,core_dict,index):
    x,y,phi_levels,srho,sphi,mask,slicemask = calc_plot(rho,phi,core_dict,index)
    mesh(x,y,srho.transpose())
    p.contour(x,y,sphi,colors='k',levels=phi_levels,antialiased=False,alpha=1.0,linewidths=1)
    p.contour(x,y,mask,levels=[0.0,0.5,1.0],colors='r',antialiased=False,alpha=0.5,linewidths=1)
    p.contour(x,y,slicemask,levels=[0.0,0.5,1.0],colors='r',antialiased=False,alpha=0.75,linewidths=2)


def plot_tree_slice(rho,phi,core_dict,bound_dict,index):
    x,y,phi_levels,srho,sphi,mask,slicemask = calc_plot(rho,phi,core_dict,index)
    x2,y2,_,_,_,mask2,slicemask2 = calc_plot(rho,phi,bound_dict,index)
    mesh(x,y,srho.transpose())
    p.contour(x,y,sphi,colors='k',levels=phi_levels,antialiased=False,alpha=1.0,linewidths=1)
    p.contour(x,y,mask,levels=[0.0,0.5,1.0],colors='r',antialiased=False,alpha=0.5,linewidths=1)
    p.contour(x2,y2,mask2,levels=[0.0,0.5,1.0],colors='b',antialiased=False,alpha=0.5,linewidths=1)
    p.contour(x2,y2,slicemask2,levels=[0.0,0.5,1.0],colors='b',antialiased=False,alpha=0.75,linewidths=2)

def plot_full_tree_bound(rho,iso_dict,bound_dict):
    surf = rho.sum(axis=2).transpose()
    x = n.arange(rho.shape[0])
    y = n.arange(rho.shape[1])
    datas = [iso_dict,bound_dict]
    colors0 = ['k','r']
    vmax0 = n.max(surf)
    p.pcolormesh(x,y,data,cmap=p.get_cmap('viridis'),vmin = 1e-3*vmax0,vmax=vmax0,norm=mco.LogNorm(),zorder=0,rasterized=True)         
    for i in range(2):
        data = ftt(rho,datas[i]).sum(axis=2).transpose()
        color = colors0[i]
        p.contour(x,y,data,origin='lower',levels=[0],colors=color,linewidths=1.0)
