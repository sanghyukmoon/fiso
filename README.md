# FISO

Fast ISOcontours



# Usage
```
from fiso.fiso_tree import construct_tree, calc_leaf
from fiso.tree_bound import compute

# Construct isocontour tree
# Phi is a numpy.ndarray of gravitational potential
iso_dict, labels, iso_list, eic_list = construct_tree(Phi)

# Calculate leaf nodes (structures that contains only 1 local minima)
leaf = calc_leaf(iso_dict, iso_list, eic_list)

# Calculate HBP and HBR dictionaries
HBP, HBR = compute(Phi, iso_dict, iso_list, eic_list)
```

# Notes to developers

## Terminology
* iso_dict : This is a dictionary containing all **structures** or **iso**s. Keys are 1D flattend indices of the structure-generating cell (the critical points of the gravitational potentials). Values correspond to the 1D flattend indices of the cells belong to those structures.

## Indexing
All cells are indexed by their flattened indices.
```
# Convert 1d flattend index to 3D (k,j,i) index
index = numpy.unravel_index(flattened_indices, shape)
```
