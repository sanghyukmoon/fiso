import numpy as n
import pylab as p
import matplotlib.colors as mco
import scipy.ndimage as sn
import sys

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
    slices = range(len(cmax))
    for i in range(len(cmax)):
        smin = max(ccenter[i]-csize,0)
        smax = min(ccenter[i]+csize,shape[i])
        slices[i] = slice(smin,smax) 
        # how to slice 3D array
        newcoords[i] = coords[i] - smin 
        # coordinates for 3-D projected mask
        slicecoords[i] = newcoords[i][coords[2] == center_coord] 
        # coords for 2-D planar mask
    return slices, newcoords, slicecoords

def plot_fiso_slice(rho,phi,core_dict,index):
    slices, newcoords, slicecoords = calc_slice(core_dict,index,rho.shape)
    center_coord = n.unravel_index(index,rho.shape)[2]
    core_phi = phi[slices][newcoords]
    # phie was in case I needed a half step in phi
    # phie = n.min(n.diff(n.sort(core_phi)))
    phi0 = n.min(core_phi)
    phi1 = n.max(core_phi) #+ phie/2.
    conres = 5.
    dphi = (phi1-phi0)/conres
    phi_levels = n.arange(phi0,phi1+conres*dphi,dphi)
    slices[2] = center_coord
    x = n.arange(slices[0].start,slices[0].stop)
    y = n.arange(slices[1].start,slices[1].stop)
    srho = rho[slices]
    sphi = phi[slices]
    mask = n.zeros(srho.shape)
    mask[newcoords[:2]] = 1.0
    slicemask = n.zeros(srho.shape)
    slicemask[slicecoords[:2]] = 1.0

    mesh(x,y,srho.transpose())
    p.contour(x,y,sphi.transpose(),colors='k',levels=phi_levels,antialiased=False,alpha=1.0,linewidths=1)
    p.contour(x,y,mask.transpose(),levels=[0.0,0.5,1.0],colors='r',antialiased=False,alpha=0.5,linewidths=1)
    p.contour(x,y,slicemask.transpose(),levels=[0.0,0.5,1.0],colors='r',antialiased=False,alpha=0.75,linewidths=2)

