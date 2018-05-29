import sys
import numpy as n
import fiso

size = 128
shape = [size]*3
x,y,z = n.indices(shape)
# set up potential with 4 point sources

def point(x0,y0,z0):
    radius = n.sqrt((x-x0)**2 + (y-y0)**2 + (z-z0)**2) + 16.
    return -1./radius

f1 = 0.25*size
f2 = 0.75*size
zlist = [0.5*size]*4
potential = point(f1,f1,zlist[0]) +\
point(f1,f2,zlist[1]) +\
point(f2,f1,zlist[2]) +\
point(f2,f2,zlist[3])

coredict,labels = fiso.find(potential)
#here, find field maxima
labels = labels.reshape(shape)
if len(sys.argv) > 1:
    exit()


import matplotlib
import matplotlib.colors as mco
matplotlib.use('agg')
import pylab as p
import fiso_contour as fc


def mesh(arr):                                                            
    p.pcolormesh(n.arange(size),n.arange(size),arr,shading='flat',edgecolors='none',norm=mco.LogNorm())  

#labels > 0
def plots(field):
    fieldz = n.max(field,axis=2)
    only_core = 1.0*field #n.ones(field.shape)
    only_core[labels < 0] = 0.0
    ocz = n.max(only_core,axis=2)
    
    p.subplot(2,2,1)
    mesh(fieldz)
    p.title('Z projection -phi')
    p.subplot(2,2,2)
    mesh(ocz)
    p.title('Only Core phi')
    phi = -field
    p.subplot(2,2,3)
    fc.plot_fiso_slice(-phi,phi,coredict,list(coredict.keys())[0])

    p.subplot(2,2,4)
    fc.plot_fiso_slice(-phi,phi,coredict,list(coredict.keys())[1])
    p.savefig('test.png')
    
plots(-potential)
