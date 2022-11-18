import time
import numpy as np
import networkx as nx
import glob
import re
import os
import itertools

import csv
from netCDF4 import Dataset

#plotting imports
import matplotlib

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import cartopy.crs as ccrs
import cartopy.feature as cfeature

import seaborn as sns
sns.set(font_scale=2.5)
sns.set_style("whitegrid")

#For cluster usage
os.chdir("/p/projects/dominoes/lena")


#self programmed code imports
import sys
sys.path.append('pycascades/modules/gen')
sys.path.append('pycascades/modules/core')

from tipping_network import tipping_network
matplotlib.use("Agg")
import pycascades as pc

from net_factory import from_nxgraph
#from gen.net_factory import from_nxgraph
from amazon import generate_network
from evolve_sde import evolve, NoEquilibrium
#from evolve import evolve, NoEquilibrium
from tipping_element import cusp
from coupling import linear_coupling
from functions_amazon import global_functions
from scipy.stats import kendalltau
from sklearn import preprocessing
import scipy.stats as stats


"This code can be used to calculate the normalized distribution of the connection strengths in the Amazon rainforest"

#sys_var = np.array(sys.argv[2:])
#year = sys_var[0]
#no_cpl_dummy = int(sys_var[1])
#adapt = float(sys_var[2])       #Range of adaptability in multiples of the standard deviation; higher adapt_fact means higher adaptability
#start_file = sys_var[3]

#year = 2004
#no_cpl_dummy = 1
adapt = 1.0
start_file = 1
year = 2005
region = 0

no_cpl_dummy = 1
#coupling = 6.0


################################GLOBAL VARIABLES################################
if no_cpl_dummy == 0: #no_cpl_dummy can be used to shut on or off the coupling; If True then there is no coupling, False with normal coupling
    no_cpl_dummy = True 
elif no_cpl_dummy == 1:
    no_cpl_dummy = False 
else:
    print("Uncorrect value given for no_cpl_dummy, namely: {}".format(no_cpl_dummy))
    exit(1)


adapt_file = np.loadtxt("probabilistic_ensemble/r_crit_start_sample_save_equal/{}.txt".format(str(start_file).zfill(3)))       #Range of adaptability in multiples of the standard deviation; higher adapt_fact means higher adaptability
adapt_fact = np.add(adapt, adapt_file)
rain_fact = 1.0   #The variable can be used to evaluate tipping points in the Amazon rainforest by artifically reducing the rain
################################################################################

#GLOBAL VARIABLES
#the first two datasets are required to compute the critical values
#Here data_crit defines the range from 73 to 2002 deifning the avarage
data_crit = np.sort(np.array(glob.glob("average_network/hydrological_gldas_average/average_both_1deg*.nc")))
data_crit_std = np.sort(np.array(glob.glob("average_network/hydrological_gldas_average/average_std_both_1deg*.nc")))

#Start with the scenario that data_eval = data_crit, to set the initial scenario to the avarage of all timeseries with minimum c-values for all nodes. 
data_eval = data_crit


###MAIN - PREPARATION###
#need changing variables from file names
dataset = data_crit[0]
net_data = Dataset(dataset)

#latlon values
lat = net_data.variables["lat"][:]
lon = net_data.variables["lon"][:]
#c = net_data.variables["c"][:]

resolution_type = "1deg"
year_type = year


lat = net_data.variables["lat"][:]
lon = net_data.variables["lon"][:]   
tuples = [(lat[idx],lon[idx]) for idx in range(lat.size)]

if region == 0:
    cells = np.loadtxt("./text_files/NWS_cells.txt", dtype=int)
    c_end = np.loadtxt("./text_files/c_end_values_{}_NWS.txt".format(year, year), usecols = (1), dtype= np.float64)
    neighbour = np.loadtxt("./text_files/neighbourslist_NWS.txt", dtype=int)
elif region == 1:
    cells = np.loadtxt("./text_files/NSA_cells.txt", dtype=int)
    c_end = np.loadtxt("./text_files/c_end_values_{}_NSA.txt".format(year,year), usecols = (1), dtype= np.float64)
    neighbour = np.loadtxt("./text_files/neighbourslist_NSA.txt", dtype=int)
elif region == 2:
    cells = np.loadtxt("./text_files/SAM_cells.txt", dtype=int)
    c_end = np.loadtxt("./text_files/c_end_values_{}_SAM.txt".format(year, year), usecols = (1), dtype= np.float64)
    neighbour = np.loadtxt("./text_files/neighbourslist_SAM.txt", dtype=int)
elif region == 3:
    cells = np.loadtxt("./text_files/NES_cells.txt", dtype=int)
    c_end = np.loadtxt("./text_files/c_end_values_{}_NES.txt".format(year, year), usecols = (1), dtype= np.float64)
    neighbour = np.loadtxt("./text_files/neighbourslist_NES.txt", dtype=int)
else:
    print(f"Whole network is selected")
    c_end = np.loadtxt("./text_files/c_end_values_{}.txt".format(year), usecols = (1), dtype= np.float64)
    neighbour = np.loadtxt("./text_files/neighbourslist.txt", dtype=int)
    cells = len(list(0, 567))

'''
if no_cpl_dummy == True:   
    cval = np.loadtxt("no_coupling/region{}/cvalues_{}_region{}.txt".format(region, year, region), dtype = np.float64)
else:
    cval = np.loadtxt("region{}/cvalues_{}_region{}.txt".format(region, year, region), dtype = np.float64)
'''

#print(f"Shape of cval is", np.shape(cval))

all_connections = np.genfromtxt("/p/projects/dominoes/lena/jobs/results/noise/coupling_strengths.txt", dtype=None)
print(all_connections)
print(np.shape(all_connections))
#tuples_NWS = [(lat, lon) for lat in range(-17, +11) for lon in range(-79, -71)]
#print(np.sort(tuples_NWS))
#tuples_NSA = [(lat, lon) for lat in range(-7, +11) for lon in range(-71, -50)]
#tuples_SAM = [(lat, lon) for lat in range(-17, -7) for lon in range(-71, -50)]
#tuples_NES = [(lat, lon) for lat in range(-17, 2) for lon in range(-50, -44)]

lat = np.unique(lat)
#print(f"Latitude values are", lat)
lon = np.unique(lon)
lat = np.append(lat, lat[-1]+lat[-1]-lat[-2]) 
lon = np.append(lon, lon[-1]+lon[-1]-lon[-2])

vals = np.empty((lat.size,lon.size)) #Vals is True or False depending on if the cells tipped
vals[:,:] = np.nan

#For different regions latitutde and longitude latitude and longitutde values
if region == 0:
    tuples_region = [(lat, lon) for lat in range(-17, +11) for lon in range(-79, -71)]                 
    #print(f"Tuple values for region 0 are:", tuples_region)
elif region == 1:
    tuples_region = [(lat, lon) for lat in range(-7, +11) for lon in range(-71, -50)]
    #print(f"Tuple values for region 0 are:", tuples_region)
elif region == 2:
    tuples_region = [(lat, lon) for lat in range(-17, -7) for lon in range(-71, -50)]
elif region == 3:
    tuples_region = [(lat, lon) for lat in range(-17, 2) for lon in range(-50, -44)]
else:
    lat = lat
    lon = lon

connection_list = []
for x,y,z in all_connections:
    if x in cells and y in cells:
        connection_list.append(tuple((x,y,z)))

sorted_list = sorted(
    connection_list,
    key=lambda t: t[2],
    reverse=True
)
print(sorted_list[:5])

#count = np.count_nonzero(sorted_list[:, -1] > 0.005)
x = [item[-1] for item in sorted_list]
x = np.array(x)
print(type(x))
a_number = 0.02
#larger_elements = [element for element in normalized if element > a_number]
#print(f"The amount of connections greater then 0.02 is", len(larger_elements))
print(f"Average connection strength is", np.mean(x))

fmt=("%d", "%d", "%1.5f")
np.savetxt("/p/projects/dominoes/lena/jobs/results/noise/coupling_strengths_region{}.txt".format(region), sorted_list, fmt=fmt)

#print(f"Precprocessed values", preprocessing.normalize(x.reshape(-1,1)))
#print(f"Precprocessed values", preprocessing.normalize(x.reshape(1,-1)))
# normalized = preprocessing.normalize(x.reshape(1,-1))
normalized = preprocessing.normalize(x.reshape(1,-1))
print(f"Normalized is", normalized.shape[1])
print(np.max(normalized))
count = np.count_nonzero(normalized >= 0.02)
print(f"Mean value of connection strength is", np.mean(normalized[:, :]))
percentage = count/normalized.shape[1]
print(f"Number greater than 0.02 is", percentage*(100))

# An "interface" to matplotlib.axes.Axes.hist() method

#0504aa
density = stats.gaussian_kde(normalized)
#n, bins, patches = plt.hist(x= [item[-1] for item in sorted_list], bins='auto', color='black',
n, bins, patches = plt.hist(x= [item for item in normalized], bins='auto', color='black',
                            alpha=0.7, rwidth=0.85, density=True)
#np.linspace(0, 0.15, 150)
plt.grid(axis='y', alpha=0.75)
plt.tick_params(axis='x', labelsize=20)
plt.tick_params(axis='y', labelsize=20)
plt.xlabel('Normalized coupling strength', fontsize = 20)
plt.ylabel('Frequency', fontsize = 20)

#plt.title('Histogramm for coupling values in region{}'.format(region))
maxfreq = n.max()
plt.xlim(0, 0.14)
plt.grid(False)
plt.margins(x=0)
plt.tight_layout()
# Set a clean upper y-axis limit.
# plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)
plt.savefig("/p/projects/dominoes/lena/jobs/results/noise/coupling_strengths_region{}.png".format(region)) #, dpi=200)
