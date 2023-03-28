# FISO

Refactored version of Fast ISOcontours developed by Alwin Mao. Original repo: https://github.com/alwinm/fiso


# Reference
[Mao, Ostriker, & Kim 2020, ApJ, 898](https://ui.adsabs.harvard.edu/abs/2020ApJ...898...52M/abstract)



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

## Wanings and TODOs
* There is an inconsistency in defining the leaf in `calc_leaf` and `compute`.
* The method to find leaf in `compute` seems to be erroneous.
* It would be much better to design a class instead of dispersed fucntions.
* `compute` does not seem to find largest possible HBR, in contrast to what is described in the method paper.

## Terminology
* iso_dict : This is a dictionary containing all **structures** or **iso**s. Keys are 1D flattend indices of the structure-generating cell (the critical points of the gravitational potentials). Values correspond to the 1D flattend indices of the cells belong to those structures.

## Indexing
All cells are indexed by their flattened indices.
```
# Convert 1d flattend index to 3D (k,j,i) index
index = numpy.unravel_index(flattened_indices, shape)
```
