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

The main scripts to build the Amazon rainforest are found in the folder Scripts/pycascades. It was etrieved from ```https://github.com/pik-copan/pycascades```. The folders Scripts/average_network and  Scripts/probabilistic_ensemble contain the data used to build the network depicting the Amazon rainforest.

# Main Scripts

The main scripts used to create the graphs in the master's thesis are listed in the folder Scripts.
An overview is given below:

- r_crit_unstable_amazon_final.py: This code can be used to get the state values as .txt files for the different experiments. Furthermore, plots of the regions of the Amazon rainforest to depict the state value distribution of the state values.
- r_crit_unstable_amazon_correlation.py: This code can be used to calculate the Moran's I coefficient depicting the correlation between neighbouring cells and plot it against the number of iterations.
-r_crit_unstable_amazon_variance.py: This code can be used to calculate the spatial variance depicting the correlation between neighbouring cells and plot it against the number of iterations.
- r_crit_unstable_amazon_final_plotting.py: This code is use to get state values when approaching the drought conditions of the years examined, before and after tip for coupling and non-coupling. Furthermore it is used to get the values of the critical functions as well as plotting the regions in terms of these dfferent critical function values.The last two bits of code can be further used to plot the state values in the regions for different coupling and noise strengths. 

For the sensitivity analysis the following scripts have been used:

- sensitivity_coupling_correlation.py and sensitivity_coupling_variance.py: These scripts were used to plot the Kendall$\tau$ values of the Moran's I coefficient and the spatial variance evolution over different coupling strengths.

- sensitivity_noise_correlation.py and sensitivity_noise_variance.py: These scripts were used to plot the Kendall$\tau$ values of the Moran's I coefficient and the spatial variance evolution over different noise strengths.

- r_crit_unstable_amazon_final_coupling_plots.py : This code is used to plot the evolution of the Moran's I coefficient and the spatial variance for different coupling strenghts

- r_crit_unstable_amazon_final_noise_plot.py : This code is used to plot the evolution of the Moran's I coefficient and the spatial variance for different noise strengths


