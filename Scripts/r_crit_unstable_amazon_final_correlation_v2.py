
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
from scipy import stats
from scipy.stats import kendalltau
import pymannkendall as mk
import pickle as pkl
"This code can be used to calculate the correlation between neighbouring cells (and plot it against average state of cell) for specially selected cells when increasing c_value for one/or more cells"
"The spatial correlation index was defined as the two-point correlation for all pairs of cells separated by distance 1, using the Moran’s coefficient (Legendre)"

#Load neighbourlist to compute correlation between cells from r_crit_unstable_amazon.py
#neighbour = np.loadtxt("./jobs/results/noise/neighbourslist.txt", dtype=int)

#Load c_values to compute correlation and variance between cells from amzon.py related to the hydro_year
# c_values = np.loadtxt("./jobs/results/noise/final/c_end_values.txt", delimiter=" ",usecols=np.arange(1, n_cols))
#c_end = np.loadtxt("./jobs/results/noise/final/c_end_values.txt", usecols = (1), dtype= np.float64)
#Get all negavtive c-values converted to 0
#c_end[c_end < 0] = 0

c_begin = np.loadtxt("./jobs/results/noise/final/c_begin_values.txt", usecols = 1, dtype = np.float64)

#c_values = np.loadtxt("./jobs/results/noise/final/c_end_values.txt", dtype=dtype)

# Choose between NWS(0), NSA(1), SAM(2), NES(3)
year = 2010
region = 0
print("Region {} selected".format(region))

if region == 0:
    cells = np.loadtxt("./jobs/results/noise/NWS_cells.txt", dtype=int)
    c_end = np.loadtxt("./jobs/results/noise/final/{}/c_end_values_{}_NWS.txt".format(year, year), usecols = (1), dtype= np.float64)
    neighbour = np.loadtxt("./jobs/results/noise/neighbourslist_NWS.txt", dtype=int)
elif region == 1:
    cells = np.loadtxt("./jobs/results/noise/NSA_cells.txt", dtype=int)
    c_end = np.loadtxt("./jobs/results/noise/final/{}/c_end_values_{}_NSA.txt".format(year,year), usecols = (1), dtype= np.float64)
    neighbour = np.loadtxt("./jobs/results/noise/neighbourslist_NSA.txt", dtype=int)
elif region == 2:
    cells = np.loadtxt("./jobs/results/noise/SAM_cells.txt", dtype=int)
    c_end = np.loadtxt("./jobs/results/noise/final/{}/c_end_values_{}_SAM.txt".format(year, year), usecols = (1), dtype= np.float64)
    neighbour = np.loadtxt("./jobs/results/noise/neighbourslist_SAM.txt", dtype=int)
elif region == 3:
    cells = np.loadtxt("./jobs/results/noise/NES_cells.txt", dtype=int)
    c_end = np.loadtxt("./jobs/results/noise/final/{}/c_end_values_{}_NES.txt".format(year, year), usecols = (1), dtype= np.float64)
    neighbour = np.loadtxt("./jobs/results/noise/neighbourslist_NES.txt", dtype=int)
else:
    print(f"Whole network is selected")
    c_end = np.loadtxt("./jobs/results/noise/final/c_end_values_{}.txt".format(year), usecols = (1), dtype= np.float64)
    neighbour = np.loadtxt("./jobs/results/noise/neighbourslist.txt", dtype=int)
    cells = range(0, 567)

c_end[ c_end < 0] = 0
t_step = 0.1
realtime_break = 100 #originally 30000 and works with 200 (see r_crt_unstable_amazon.py)
timesteps = (realtime_break/t_step)
dc = (c_end/timesteps)

os.chdir("/p/projects/dominoes/lena/jobs/results/noise/final/{}".format(year))

c_lis = []


#Load no_coupling and coupling variables for given year and region
ilist_nocpl = np.loadtxt("no_coupling/i_list_{}_region{}.txt".format(year, region), dtype = int)
all_states_nocpl = np.loadtxt("no_coupling/states_cusps_{}_region{}.txt".format(year, region), dtype = np.float64)
ilist = np.loadtxt("i_list_{}_region{}.txt".format(year, region), dtype = int)
all_states = np.loadtxt("states_cusps_{}_region{}.txt".format(year, region), dtype = np.float64)

#Try for region 0 with 2000 timesteps
#ilist_nocpl = np.loadtxt("no_coupling/i_list_{}_region{}_2000.txt".format(year, region), dtype = np.float64)
#all_states_nocpl = np.loadtxt("no_coupling/states_cusps_{}_region{}_2000.txt".format(year, region), dtype = np.float64)
#ilist = np.loadtxt("i_list_{}_region{}_2000.txt".format(year, region), dtype = np.float64)
#all_states = np.loadtxt("states_cusps_{}_region{}_2000.txt".format(year, region), dtype = np.float64)

#ilist = [i/100 for i in ilist]  
#ilist_nocpl = [i/100 for i in ilist_nocpl]
print(ilist)

#Try out with another sum approach
def correlation():
    cor =  []
    for iteration in range(0, len(ilist)-1):
        # print(f"Iteration is", iteration)
        sum = 0
        den_list = []
        avg = np.mean(all_states[:, iteration])
        for i in range(0, len(list(cells))):
            for j in range(0, len(list(cells))):
                cell_pair = [cells[i], cells[j]]
                if (cell_pair == neighbour).all(1).any() == True:
                    sum = sum + ((all_states[i, iteration] - avg)*(all_states[j, iteration] - avg))
                    #print(cells[i], cells[j])
                    #print(f"Sum is", sum)
        
        #Calculate denominator for 2-point-correlation
        for x in range(0, len(list(cells))):
            den_item = np.square(all_states[x, iteration] - avg)
            den_list.append(den_item)

        #Calculate correlation for every iteration
        corr_item = len(list(cells)) * (sum) /(len(list(neighbour)) * np.sum(den_list))
        cor.append(corr_item)
    return cor

def correlation_nocpl():
    cor_nocpl =  []
    for iteration in range(0, len(ilist_nocpl)-1):
        #print(f"Iteration is", iteration)
        sum = 0
        den_list = []
        avg = np.mean(all_states_nocpl[:, iteration])
        for i in range(0, len(list(cells))):
            for j in range(0, len(list(cells))):
                cell_pair = [cells[i], cells[j]]
                if (cell_pair == neighbour).all(1).any() == True:
                    sum = sum + ((all_states_nocpl[i, iteration] - avg)*(all_states_nocpl[j, iteration] - avg))
                    #print(cells[i], cells[j])
                    #print(f"Sum is", sum)
        
        #Calculate denominator for 2-point-correlation
        for x in range(0, len(list(cells))):
            den_item = np.square(all_states_nocpl[x, iteration] - avg)
            den_list.append(den_item)

        #Calculate correlation for every iteration
        corr_item = len(list(cells)) * (sum) /(len(list(neighbour)) * np.sum(den_list))
        cor_nocpl.append(corr_item)
    return cor_nocpl


cor_cpl = correlation()
cor_no_cpl = correlation_nocpl()


# Print out tau kendall value for the whole time series
mk_cpl = mk.original_test(cor_cpl)
mk_nocpl = mk.original_test(cor_no_cpl)
print(f"Mk is", mk_cpl)
print(f"Mk no coupling is", mk_nocpl)
kend_item_cpl = round(mk_cpl.Tau,2)
kend_item_nocpl = round(mk_nocpl.Tau,2)
kend_items = [kend_item_cpl, kend_item_nocpl]
kend_item_cpl_only = kendalltau(ilist[:-1], cor_cpl)
kend_item_nocpl_only = kendalltau(ilist_nocpl[:-1], cor_no_cpl)
print(f"Kendalltauitem coupling", kend_item_cpl_only)
print(f"Kendalltau item no coupling", kend_item_nocpl_only)

'''
cor_no_cpl = np.loadtxt("noise_states/no_coupling/region{}/correlation_values/correlation{}_region{}_coupling0.01.txt".format(region, year, region), dtype = np.float64)
cor_cpl = np.loadtxt("noise_states/region{}/correlation_values/correlation{}_region{}_coupling0.01.txt".format(region, year, region), dtype = np.float64)
mk_cpl = mk.original_test(cor_cpl)
mk_nocpl = mk.original_test(cor_no_cpl)
kend_item_cpl = round(mk_cpl.Tau,2)
kend_item_nocpl = round(mk_nocpl.Tau,2)
'''
#Plotting squences for variance and correlation

fig = plt.figure(figsize = (8,6))
ax1 = fig.add_subplot(111)


#Load coupling and no_coupling values for tipped cells
line1, =  ax1.plot(ilist[:-1], cor_cpl, 'royalblue', linestyle = "-") # coupling
line2, = ax1.plot(ilist_nocpl[:-1], cor_no_cpl, 'maroon', linestyle = "--") #no coupling

ax2 = ax1.twinx()
plt.tick_params(right=False, bottom=True)

#ax1.set_yticks([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]) 
ax1.set_ylim(-1.0, 1.0)
ax1.set_xticks([0, 3, 6, 9, 12, 15])
#x1.set_xlim(0, 61)

if region == 0: 
    pass
else:
    ax1.axes.yaxis.set_ticklabels([])  #Disbale for first image

ax2.axes.yaxis.set_ticklabels([])
ax3 = ax1.twiny()
ax3.axes.xaxis.set_ticklabels([])
ax3.tick_params(top=False)


#ax1.set_xlabel('Iterations')

#ax1.set_ylabel('Spatial variance', color = 'black', fontsize=17, labelpad=15)
ax1.tick_params(axis='x', labelsize = 25)
ax1.tick_params(axis='y', labelsize = 25)
ax1.margins(x=0)
ax1.grid(False)
ax2.grid(False)
ax3.grid(False)


ax1.xaxis.label.set_size(17)
ax1.yaxis.label.set_size(17)

#lengend1 = plt.legend(kend_items, ['Kendtall-tau', 'Kendall-tau no coupling'], loc = 'center left')
print(f"P-value rounded", round(mk_cpl.p,2))
print(f"P-value no coupling", round(mk_nocpl.p,2))

if float(round(mk_cpl.p,2)) <= float(0.05):
    textstr = ', '.join((
        r'$\mathrm{Kendall \tau}=%.2f$' % (kend_item_cpl, ),
        r'$\mathrm{p}<%.2f$' % (0.05, )))
else:
    textstr = ', '.join((
        r'$\mathrm{Kendall \tau}=%.2f$' % (kend_item_cpl, ),
        r'$\mathrm{p}>%.2f$' % (0.05, )))

if float(round(mk_nocpl.p,2)) <= float(0.05):
    textstr_nocpl = ', '.join((
        r'$\mathrm{Kendall \tau}=%.2f$' % (kend_item_nocpl, ),
        r'$\mathrm{p}<%.2f$' % (0.05, )))
else:
    textstr_nocpl = ', '.join((
        r'$\mathrm{Kendall \tau}=%.2f$' % (kend_item_nocpl, ),
        r'$\mathrm{p}>%.2f$' % (0.05, )))

'''
if region == 0 or region == 1:
    ax1.text(0.03, 0.6, textstr, transform=ax1.transAxes, fontsize=23, color='royalblue')  #Doesnt fit for region2/3 when plotting correlation
    ax1.text(0.03, 0.53, textstr_nocpl, transform=ax1.transAxes, fontsize=23, color='maroon')
elif region == 3 and year == 2007:
    ax1.text(0.35, 0.20, textstr, transform=ax1.transAxes, fontsize=23, color='royalblue')
    ax1.text(0.35, 0.13, textstr_nocpl, transform=ax1.transAxes, fontsize=23, color='maroon')
else:
    ax1.text(0.25, 0.20, textstr, transform=ax1.transAxes, fontsize=23, color='royalblue')
    ax1.text(0.25, 0.13, textstr_nocpl, transform=ax1.transAxes, fontsize=23, color='maroon')
'''

#For version 3 
#if region == 0 or region == 1:
ax1.text(0.05, 0.1, textstr, transform=ax1.transAxes, fontsize=25, color='royalblue')  #Doesnt fit for region2/3 when plotting correlation
ax1.text(0.05, 0.03, textstr_nocpl, transform=ax1.transAxes, fontsize=25, color='maroon')
#else:
#    ax1.text(0.25, 0.20, textstr, transform=ax1.transAxes, fontsize=23, color='royalblue')
#    ax1.text(0.25, 0.13, textstr_nocpl, transform=ax1.transAxes, fontsize=23, color='maroon')

ax1.text(0.05, 0.95,'i)',
     horizontalalignment='center',
     verticalalignment='center',
     transform = ax1.transAxes,
     fontsize=25,
     weight='bold')

if region == 0 and int(year) == 2005:
    plt.legend((line1,line2), ('Coupling','No coupling'), loc='upper right', fontsize=25)
else:
    pass
#plt.legend((line1,line2), ('Coupling','No coupling'), loc='upper right', fontsize=20)
#plt.title("Spatial variance approaching year {} drought scenario for region {}".format(year, region), fontsize=10)
ax1.axvline(x=ilist[-2], color='grey') 
#ax1.axvline(x=ilist_nocpl[-2], linestyle = "--", color='grey') 
plt.tight_layout()

fig.savefig("newaxis_correlation_unstable_amaz_{}_final_region{}_2000.png".format(year, region), dpi=200)
