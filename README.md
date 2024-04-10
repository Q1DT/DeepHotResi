
## Sequence-based Deep Learning Framework for Identifying Hotspot Residues in Protein-RNA Complexes

## Table of Contents: 

- [Description](#description)
- [System and hardware requirements](#system-and-hardware-requirements)
- [Software prerequisites](#software-prerequisites)

## Description

This study has developed DeepHotResi, a deep learning framework for predicting hotspot residues in protein-
RNA complexes. The method integrates contact maps from protein structures, sequence data, and features derived
from the esmfold model, employing Graph Attention Networks for accurate prediction. To address data limitations, a
new, larger dataset was employed, significantly improving the model’s effectiveness. The experimental results show that
DeepHotResi can effectively identify hotspots in protein-RNA complexes, and it outperforms existing methods.

## System and hardware requirements

DeepHotResi has been tested on Linux (Intel(R) Core(TM) i9-9900K CPU @ 3.60GHz 
processesor and 16GB of memory allotment)

## Software prerequisites 
The following is the list of required libraries and programs, as well as the version on which it was tested (in parenthesis).
* [Python](https://www.python.org/) (3.6)
* [ESMFold](https://github.com/facebookresearch/esm) . (esm2_t36_3B_UR50D)
* [BioPython](https://github.com/biopython/biopython) .
* [DSSP](https://github.com/cmbi/dssp) . (2.3.0)
* [DGL](https://www.dgl.ai/). (0.6.0). 
* [CD-HIT](https://github.com/weizhongli/cdhit/releases). (4.8.1) 
* [Pymol](https://pymol.org/2/). This optional plugin allows one to visualize surface files in PyMOL.
* [torch](https://pytorch.org/). (1.9.0) 
* [torch_geometric](https://pytorch.org/). (2.3.1) 
* [torchvision](https://pytorch.org/). (0.10.0+cu111) 
* pandas. (2.0.1) 

## Datasets

| FILE NAME            | DESCRIPTION                                                   |
|----------------------|---------------------------------------------------------------|
| data_dict.pkl        | Convert data to dictionary format.(['Protein Name', 'Sequence', 'Label'])                           |
| data_dict_test.pkl   | Same as above, but prepared for Test set.                           |
| protein_dict_test.pkl| 3D coordinate data of protein CA atoms                                                |





## Feature

| FEATURE NAME        | DESCRIPTION                                                       |
|---------------------|-------------------------------------------------------------------|
| DSSP                | Secondary structure and solvent accessibility annotation.        |
| ESM                 | Evolutionary context through deep learning embeddings.           |
| PSSM_npy            | Sequence homology and conservation via scoring matrices.         |
| Distance Matrices   | Spatial distances between residues for 3D structure insight.      |
| HMM                 | Statistical properties and functional sites of protein families. |
