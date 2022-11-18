

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

"This code can be used to calculate the variance (and plot it against average state of cell) and the respective Kendall-Tau for different noise levels"

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
year = 2005
region = 3

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

os.chdir("/p/projects/dominoes/lena/jobs/results/noise/final/{}/noise_states".format(year))

c_lis = []

#Empyty lists to append var_items and cor_items    

kendall_tau_cpl = []
p_cpl = []
kendall_tau = []
p = []

noise = [0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05, 0.055, 0.06, 0.065, 0.07, 0.075, 0.08, 0.085, 0.09, 0.095, 0.1]

#Calculate variance from loaded states for every noise coupling and no coupling
def variance():

    for item in noise:
        var_nocpl = []
        var = []
        print(f"Item in noise is", item)
        all_states_nocpl = np.loadtxt("no_coupling/region{}/states_cusps_{}_region{}_noise{}.txt".format(region, year, region, item), dtype = np.float64)
        all_states = np.loadtxt("region{}/states_cusps_{}_region{}_noise{}.txt".format(region, year, region, item), dtype = np.float64)
        print(f"Shape of all states is :", np.shape(all_states_nocpl))
        print(f"Shape of all states coupling is", np.shape(all_states))
        #if all_states.shape[1] or all_states_nocpl.shape[1] <= 1:
        #    num_cols_cpl = 1
        #    num_cols = 1
        #else:
        num_cols_cpl = all_states.shape[1]
        print(f"num of columns for cupling are:", num_cols_cpl)
        num_cols = all_states_nocpl.shape[1]

        for i in range(0, num_cols-1): #Don't include column with tipped cells
            #item_squared = np.sum(np.square(all_states_nocpl[:,i]))
            item_squared = np.mean(np.square(all_states_nocpl[:,i]))
            print(f"item squared for no coupling looks like", item_squared)
            #item_summed = np.sum(all_states_nocpl[:,i])
            item_summed = np.square(np.mean(all_states_nocpl[:, i]))
            print(f"item summed for no coupling looks like", item_summed)
            #item_var = (item_squared - item_summed) / (len(list(cells))**2)
            item_var = item_squared - item_summed
            print(f"item var looks like", item_var)
            var_nocpl.append(item_var)

        path =  '/p/projects/dominoes/lena/jobs/results/noise/final/{}/noise_states'.format(year, region)
        fmt = '%1.5f'
        np.savetxt(os.path.join(path, "no_coupling/region{}/variance{}_region{}_coupling{}.txt".format(region, year, region, item)), var_nocpl, fmt = fmt)

        mk_nocpl = mk.original_test(var_nocpl)  # The ml.original_test() gives out same value for Kendall$\tau$ and p- as for kendalltau(var_nocpl, iterations) or kendalltau(var_no_cpl, noise)
        print(f"Mk no coupling is", mk_nocpl)
        kend_item_nocpl, pval_nocpl = mk_nocpl.Tau, mk_nocpl.p
        kendall_tau.append(kend_item_nocpl)
        print(f"Shape of kendall_tau_no_cpl is_", np.shape(kendall_tau))
        p.append(pval_nocpl)
        

        for j in range(0, num_cols_cpl-1):
            #item_squared = np.sum(np.square(all_states[:,j]))
            item_squared = np.mean(np.square(all_states[:,j]))
            #print(f"item squared for coupling looks like", item_squared)
            item_summed = np.square(np.mean(all_states[:, j]))
            #item_summed = np.square(np.sum(all_states[:,j]))
            #print(f"item summed for coupling looks like", item_summed)
            #item_var = (item_squared - item_summed) / (len(list(cells))**2)
            item_var = item_squared - item_summed
            #print(f"item var for coupling looks like", item_var)
            var.append(item_var)
        print(f"Shape of var is", np.shape(var))

        path =  '/p/projects/dominoes/lena/jobs/results/noise/final/{}/noise_states'.format(year, region)
        fmt = '%1.5f'
        np.savetxt(os.path.join(path, "region{}/variance{}_region{}_coupling{}.txt".format(region, year, region, item)), var, fmt = fmt)

        #kendall_tau_only = kendalltau(np.mean(var), noise)
        #print(f"kendall tau only is", kendall_tau_only)
        mk_cpl = mk.original_test(var)
        print(f"MK coupling is", mk_cpl)
        kend_item_cpl, pval_cpl = mk_cpl.Tau, mk_cpl.p
        kendall_tau_cpl.append(kend_item_cpl)
        print(f"Shape of kendall_tau_no_cpl is_", np.shape(kendall_tau_cpl))
        p_cpl.append(pval_cpl)

    path =  '/p/projects/dominoes/lena/jobs/results/noise/final/{}/noise_states'.format(year, region)
    fmt = '%1.2f'
    # fmt = '%d'
    np.savetxt(os.path.join(path, "kendalltau_{}_region{}_noise.txt".format(year, region)), (kendall_tau_cpl, p_cpl), fmt = fmt, delimiter=" ")
    np.savetxt(os.path.join(path, "no_coupling/kendalltau_{}_region{}_noise.txt".format(year, region)), (kendall_tau, p),  fmt = fmt, delimiter=" ")

    return kendall_tau, p, kendall_tau_cpl, p_cpl

info_var = variance()
kendall_no_cpl = info_var[0]
pval_nocpl = info_var[1]
kendall_cpl = info_var[2]
pval_cpl = info_var[3]
print(f"Kendall cpl  is", kendall_cpl)

print(f"noise is", noise)
#print(f"noise 14th value is", noise[14])
#Plotting squences for variance and correlation

fig = plt.figure(figsize = (8,6))
ax1 = fig.add_subplot(111)
ax1.set_ylim(-1.0,1.0)

#Load coupling and no_coupling values for tipped cells
#line1, =  ax1.plot(ilist, cpl_cell, 'black', linestyle = "-") # coupling
#line2, = ax1.plot(ilist_nocpl, no_cpl_cell, 'black', linestyle = "--") #no coupling
line1, =  ax1.plot(noise, kendall_cpl, 'black', linestyle = "-") # coupling
line2, = ax1.plot(noise, kendall_no_cpl, 'black', linestyle = "--") #no coupling

#ax1.set_xlabel('Noise strength')
#ax1.set_ylabel('Kendall {} rank correlation'.format(r'$\mathrm{\tau}$'), color = 'black')
ax1.tick_params(axis='x', labelsize = 25)
ax1.tick_params(axis='y', labelsize = 25)


#From plot of sensitiviyty
ax2 = ax1.twinx()
plt.tick_params(right=False, bottom=True)
ax1.set_yticks([-1.0, -0.5, 0.0, 0.5, 1.0])
#ax2.set_ylim(0, 0.01)
#ax1.set_xticks([0,0.005,0.01,0.015,0.02,0.025,0.0]) 
#ax1.set_ylim(0, 0.01)
if region == 0:
    pass
else:
    ax1.axes.yaxis.set_ticklabels([]) #Disbale for first image

ax2.axes.yaxis.set_ticklabels([])
ax2.axes.xaxis.set_ticklabels([])
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


#ax2 = ax1.twinx()

#line3, = ax2.plot(ilist, var_cpl, 'g')
#line4, = ax2.plot(ilist_nocpl, var_no_cpl, 'g', linestyle = "--")
'''
textstr = '\n'.join((
    r'$\mathrm{Kendall-tau}=%.2f$' % (kend_item_cpl, ),
    r'$\mathrm{Kendall-tau(nocpl)}=%.2f$' % (kend_item_nocpl, )))

ax1.text(0.01, 0.7, textstr, transform=ax1.transAxes, fontsize=8)
#bbox=dict(facecolor='black', alpha=0.5),
'''
if region == 0:
    plt.legend((line1, line2), ('Coupling', 'No coupling'), loc='lower left', fontsize=25)
else:
    pass

#ax2.set_ylabel('Spatial variance', color = 'g')
#ax2.tick_params(axis='y', labelsize = 8)
#ax2.set_ylim(0, 0.01)

axes = plt.gca()
#axes.ticklabel_format(style='plain', axis='y', scilimits=(0,0))
ax1.xaxis.label.set_size(25)
ax1.yaxis.label.set_size(25)
#ax2.yaxis.label.set_size(10)  

ax1.text(0.95, 0.95,'d)',
     horizontalalignment='center',
     verticalalignment='center',
     transform = ax1.transAxes,
     fontsize=25,
     weight='bold')

#plt.title("Varying c-values for cusps 468, 487 and 505, selected network upper right (0.01*60 rate)", fontsize=10)
#plt.title("Kendall-Tau Rank variation for different noise values approaching year {} drought scenario for region {}".format(year, region), fontsize=10)
plt.tight_layout()
#plt.gca().add_artist(lengend1)

#if no_cpl_dummy == True:
#    fig.savefig("no_coupling/spat_var_unstable_amaz_{}_{}_adaptsample{}_adaptsigma{}_{}_{}_noise{}_std1.png".format(resolution_type, 
#        year_type, str(start_file).zfill(3), int(np.around(100*adapt)), id, float(rain_fact), noise), dpi=200)
#else:

fig.savefig("noise_sensitivity_{}_region_{}_variance_v2.png".format(year, region), dpi=200)
