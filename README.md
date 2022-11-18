# Spatial early warning signal detection on a complex network of the Amazon rainforest
Pyhtonframework to study the detection of SEWSs on the Amazon rainforest

# Installation

```
conda create -n SEWS python=3.9
conda deactivate 
conda activate SEWS 
conda install -c conda-forge mamba 
mamba install -c conda-forge numpy scipy matplotlib cartopy seaborn netCDF4 networkx ipykernel 
git clone https://github.com/lenas95/SEWS_Amazon_rainforest 
cd SEWS 
pip install -e . 
```
# Description

The main scripts to build the Amazon rainforest are found in the folder amazon_rainforest and pycascades. Both were retrieved from ```https://github.com/pik-copan/pycascades```
