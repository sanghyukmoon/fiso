import numpy as n
import pylab as p

import fiso.ext.fiso_tree as tree
tree.fiso.boundary_mode = 'wrap' # makes boundaries periodic

import fiso.tools.contour as ftc
import fiso.tools.tools as ftt
import fiso.tools.tree_bound as ftb

from athena_read import athdf

filename = 'Cloud.out2.00040.athdf'


athena = athdf(filename)

rho = athena['dens'][:]
phi = athena['phi'][:]
mom1 = athena['mom1'][:]
mom2 = athena['mom2'][:]
mom3 = athena['mom3'][:]


iso_dict,iso_label,iso_list,eic_list = tree.find(phi)

# ftc.plot_tree_outline(rho,iso_dict) #

# Calculate leaf nodes from full tree
leaf_dict = ftt.calc_leaf(iso_dict,iso_list,eic_list)

def plot(iso_dict):
    # Density, but only including iso_dict
    rho_iso = ftt.filter_dict(rho,iso_dict)

    x = n.arange(rho.shape[0])
    y = n.arange(rho.shape[1])
    ftc.mesh(x,y,ftc.surface(rho_iso))
    



# In this example data there is no pressure or magnetic field so set them to 0.0
pressure = 0.0*rho
bpressure = 0.0*rho
velx = mom1/rho
vely = mom2/rho
velz = mom3/rho
data = [rho,phi,pressure,bpressure,velx,vely,velz]

# In my paper I called them HPR and HBR
# HPR is the isocontour and HBR is the bound cells
hpr_dict, hbr_dict = ftb.compute(data,iso_dict,iso_list,eic_list)


nh = 2
nw = 3
ni = 0
ni += 1
p.subplot(nh,nw,ni)
plot(iso_dict)
p.title("Iso")
ni += 1
p.subplot(nh,nw,ni)
plot(leaf_dict)
p.title("Leaf")
ni += 1
p.subplot(nh,nw,ni)
plot(hpr_dict)
p.title("Parent of bound")
ni += 1
p.subplot(nh,nw,ni)

plot(hbr_dict)
p.title("Bound")

ni += 1
p.subplot(nh,nw,ni)
ftc.plot_full_tree_bound(rho,iso_dict,hbr_dict)

p.show()
