# Shear Periodic Boundary Conditions
# X, Y periodic.
# Z clip

# mem version takes less memory but will take 20% more time per cell

from fiso import fiso
import fiso.boundary.shear_periodic as boundary
import numpy as np

cell_shear = 0
# add shear to Y when X = X_min, X_max
def setup(data,cut):
    '''
    Setup for fiso find algorithms. 
    Computes local minima, sort, cutoff, neighbors.

    Arguments:
    data: n-D NumPy array
    cut: if float, compute cutoff for finder so it does not explore data > cut

    Also depends on:
    fiso.ext.fiso_sp.cell_shear: integer, default 0
    fiso.boundary.shear_periodic globals (not fiso globals)
    '''
    dshape = data.shape
    dlist = data.reshape(-1) #COPY
    order = dlist.argsort() #an array of real index locations #COPY
    fiso.timer('sort')
    cutoff = len(order) #number of cells to process
    #optional cutoff
    if type(cut) is float:
        cutoff = np.searchsorted(dlist[order],cut)

    # precompute neighbor indices
    bi, bpcn, pcn = boundary.make_pcn_mem(dshape,cell_shear)
    fiso.timer('precompute neighbor indices')

    # find minima without bc
    # find minima with bc
    # combine
    mfw0 = np.where(fiso.find_minima_no_bc(data).reshape(-1))[0]
    mfw1 = fiso.find_minima_boundary_only(dlist,bi,bpcn)
    mfw = np.unique(np.sort(np.append(mfw0,mfw1)))
    return mfw,order,cutoff,pcn

fiso.setup = setup
find = fiso.find
