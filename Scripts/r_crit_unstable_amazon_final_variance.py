

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

"This code can be used to calculate the variance (and plot it against average state of cell) for specially selected cells when increasing c_value for one/or more cells"

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

os.chdir("/p/projects/dominoes/lena/jobs/results/noise/final/{}".format(year))

c_lis = []

#Empyty lists to append var_items and cor_items    
var_nocpl = []
var = []

#Load no_coupling and coupling variables for given year and region
ilist_nocpl = np.loadtxt("no_coupling/i_list_{}_region{}.txt".format(year, region), dtype = int)
all_states_nocpl = np.loadtxt("no_coupling/states_cusps_{}_region{}.txt".format(year, region), dtype = np.float64)
ilist = np.loadtxt("i_list_{}_region{}.txt".format(year, region), dtype = int)
all_states = np.loadtxt("states_cusps_{}_region{}.txt".format(year, region), dtype = np.float64)
print(f"Shape of all states is :", np.shape(all_states_nocpl))
print(f"Shape of ilistnocpl is:", np.shape(ilist_nocpl))
print(f"Shape of all states coupling is", np.shape(all_states))
print(f"Shape of i list is", np.shape(ilist))

#Calculate variance from loaded states
def variance():

    for i in range(0, len(ilist_nocpl)-1):
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

    for j in range(0, len(ilist)-1):
        #item_squared = np.sum(np.square(all_states[:,j]))
        item_squared = np.mean(np.square(all_states[:,j]))
        print(f"item squared for coupling looks like", item_squared)
        item_summed = np.square(np.mean(all_states[:, j]))
        #item_summed = np.square(np.sum(all_states[:,j]))
        print(f"item summed for coupling looks like", item_summed)
        #item_var = (item_squared - item_summed) / (len(list(cells))**2)
        item_var = item_squared - item_summed
        print(f"item var for coupling looks like", item_var)
        var.append(item_var)

    return var_nocpl, var

info_var = variance()
var_no_cpl = info_var[0]
var_cpl = info_var[1]

cpl_cell = np.loadtxt("cusp_{}_region{}.txt".format(year, region), dtype=np.float64)
no_cpl_cell = np.loadtxt("no_coupling/cusp_{}_region{}.txt".format(year, region), dtype = np.float64)

# Print out tau kendall value for the whole time series
mk_cpl = mk.original_test(var_cpl)
mk_nocpl = mk.original_test(var_no_cpl)
print(f"Mk is", mk_cpl)
print(f"Mk no coupling is", mk_nocpl)
kend_item_cpl = round(mk_cpl.Tau,2)
p_item_cpl = mk_cpl.p
print(p_item_cpl)
kend_item_nocpl = round(mk_nocpl.Tau,2)
p_item_nocpl = mk_nocpl.p
print(p_item_nocpl)
kend_items = [kend_item_cpl, kend_item_nocpl]
kend_item_cpl_only = kendalltau(var_cpl, ilist[:-1])
kend_item_nocpl_only = kendalltau(var_no_cpl, ilist_nocpl[:-1])
print(f"Kendalltauitem coupling", kend_item_cpl_only)
print(f"Kendalltau item no coupling", kend_item_nocpl_only)

#Plotting squences for variance and correlation

fig = plt.figure(figsize = (8,6))
ax1 = fig.add_subplot(111)
#ax1.set_ylim(-1.0,1.0)

#Load coupling and no_coupling values for tipped cells
#line1, =  ax1.plot(ilist, cpl_cell, 'black', linestyle = "-") # coupling
#line2, = ax1.plot(ilist_nocpl, no_cpl_cell, 'black', linestyle = "--") #no coupling

line1, = ax1.plot(ilist[:-1], var_cpl, 'royalblue')
line2, = ax1.plot(ilist_nocpl[:-1], var_no_cpl, 'maroon', linestyle = "--")

#ax1.set_xlabel('Timeseries')
#ax1.set_ylabel('Average state for tipped cells', color = 'black')
#ax1.tick_params(axis='x', labelsize = 13)
#ax1.tick_params(axis='y', labelsize = 13)

#ax2 = ax1.twinx()

#line3, = ax2.plot(ilist, var_cpl, 'royalblue')
#line4, = ax2.plot(ilist_nocpl, var_no_cpl, 'maroon', linestyle = "--")

#textstr = '\n'.join((
#    r'$\mathrm{Kendall-\tau}=%.2f, {p-value}$' % (kend_item_cpl, p_item_cpl),
#    r'$\mathrm{Kendall-\tau}=%.2f, {p-value}$' % (kend_item_nocpl, p_item_nocpl)))

textstr = ', '.join((
    r'$\mathrm{Kendall \tau}=%.2f$' % (kend_item_cpl, ),
    r'$\mathrm{p}<%.2f$' % (0.05, )))

textstr_nocpl = ', '.join((
    r'$\mathrm{Kendall \tau}=%.2f$' % (kend_item_nocpl, ),
    r'$\mathrm{p}<%.2f$' % (0.05, )))

ax1.text(0.01, 0.7, textstr, transform=ax1.transAxes, fontsize=13, color='royalblue')
ax1.text(0.01, 0.65, textstr_nocpl, transform=ax1.transAxes, fontsize=13, color='maroon')
#bbox=dict(facecolor='black', alpha=0.5),

'''
if year == 2005:
    if region == 0:
        plt.legend((line1, line2, line3, line4), 
        ('Average state for cell 47', 'Average state for cell 47 no coupling', 'Spatial variance', 'Spatial variance no coupling'), prop={'size': 8}, loc='upper left')
    elif region == 1:
        plt.legend((line1, line2, line3, line4), 
        ('Average state for cell 172', 'Average state for cell 172 no coupling', 'Spatial variance', 'Spatial variance no coupling'), prop={'size': 8}, loc='upper left')
    elif region == 2:
        plt.legend((line1, line2, line3, line4), 
        ('Average state for cell 440', 'Average state for cell 440 no coupling', 'Spatial variance', 'Spatial variance no coupling'), prop={'size': 8}, loc='upper left')
    elif region == 3:
        plt.legend((line1, line2, line3, line4), 
        ('Average state for cell 538', 'Average state for cell 538 no coupling', 'Spatial variance', 'Spatial variance no coupling'), prop={'size': 8}, loc='upper left')
    else:
        plt.legend((line1, line2, line3, line4), 
        ('Average state for cell 47', 'Average state for cell 47 no coupling', 'Spatial variance', 'Spatial variance no coupling'), prop={'size': 8}, loc='upper left')

elif year == 2007:
    if region == 0:
        plt.legend((line1, line2, line3, line4), 
        ('Average state for cell 30', 'Average state for cell 30 no coupling', 'Spatial variance', 'Spatial variance no coupling'), prop={'size': 8}, loc='upper left')
    elif region == 1:
        plt.legend((line1, line2, line3, line4), 
        ('Average state for cell 218', 'Average state for cell 218 no coupling', 'Spatial variance', 'Spatial variance no coupling'), prop={'size': 8}, loc='upper left')
    elif region == 2:
        plt.legend((line1, line2, line3, line4), 
        ('Average state for cell 400', 'Average state for cell 400 no coupling', 'Spatial variance', 'Spatial variance no coupling'), prop={'size': 8}, loc='upper left')
    elif region == 3:
        plt.legend((line1, line2, line3, line4), 
        ('Average state for cell 540', 'Average state for cell 540 no coupling', 'Spatial variance', 'Spatial variance no coupling'), prop={'size': 8}, loc='upper left')
    else:
        plt.legend((line1, line2, line3, line4), 
        ('Average state for cell 30', 'Average state for cell 30 no coupling', 'Spatial variance', 'Spatial variance no coupling'), prop={'size': 8}, loc='upper left')
else:
    if region == 0:
        plt.legend((line1, line2, line3, line4), 
        ('Average state for cell 30', 'Average state for cell 30 no coupling', 'Spatial variance', 'Spatial variance no coupling'), prop={'size': 8}, loc='upper left')
    elif region == 1:
        plt.legend((line1, line2, line3, line4), 
        ('Average state for cell 216', 'Average state for cell 216 no coupling', 'Spatial variance', 'Spatial variance no coupling'), prop={'size': 8}, loc='upper left')
    elif region == 2:
        plt.legend((line1, line2, line3, line4), 
        ('Average state for cell 440', 'Average state for cell 440 no coupling', 'Spatial variance', 'Spatial variance no coupling'), prop={'size': 8}, loc='upper left')
    elif region == 3:
        plt.legend((line1, line2, line3, line4), 
        ('Average state for cell 538', 'Average state for cell 538 no coupling', 'Spatial variance', 'Spatial variance no coupling'), prop={'size': 8}, loc='upper left')
    else:
        plt.legend((line1, line2, line3, line4), 
        ('Average state for cell 30', 'Average state for cell 30 no coupling', 'Spatial variance', 'Spatial variance no coupling'), prop={'size': 8}, loc='upper left')
'''

#ax2.set_ylabel('Spatial variance', color = 'g')
#ax2.tick_params(axis='y', labelsize = 8)
#ax2.set_ylim(0, 0.01)

#ax1.set_ylabel('Spatial variance', color = 'black')
#ax1.tick_params(axis='y', labelsize = 12, direction='in')
ax1.set_ylim(0, 0.01)


axes = plt.gca()
#axes.ticklabel_format(style='plain', axis='y', scilimits=(0,0))
ax1.xaxis.label.set_size(13)
ax1.set_yticks([0,0.001,0.002,0.003,0.004,0.005,0.006,0.007,0.008,0.009,0.01])
ax1.tick_params(axis='both', labelsize = 12, direction='in')

#ax1.axes.get_yaxis().set_ticks(([]))
plt.grid(False)
plt.yticks(color='black')
#ax1.yaxis.label.set_size(13)
#ax2.yaxis.label.set_size(10)

#Load ilist for first cell tipped when plotting sholw network
if region == 4:
    ilist_first0 = np.loadtxt("i_list_{}_region0.txt".format(year), dtype = int)
    #ilist_first1 = np.loadtxt("i_list_{}_region1.txt".format(year), dtype = int)
    #ilist_first2 = np.loadtxt("i_list_{}_region2.txt".format(year), dtype = int)
    #ilist_first3 = np.loadtxt("i_list_{}_region3.txt".format(year), dtype = int)
    ilist_firstnocpl0 = np.loadtxt("no_coupling/i_list_{}_region0.txt".format(year), dtype = int)
    #ilist_firstnocpl1 = np.loadtxt("no_coupling/i_list_{}_region1.txt".format(year), dtype = int)
    #ilist_firstnocpl2 = np.loadtxt("no_coupling/i_list_{}_region2.txt".format(year), dtype = int)
    #ilist_firstnocpl3 = np.loadtxt("no_coupling/i_list_{}_region3.txt".format(year), dtype = int)
    plt.axvline(x=ilist_first0[-1]) 
    #plt.axvline(x=ilist_first1[-1])
    #plt.axvline(x=ilist_first2[-1])
    #plt.axvline(x=ilist_first3[-1])
    plt.axvline(x=ilist_firstnocpl0[-1], linestyle = "--")
    #plt.axvline(x=ilist_firstnocpl1[-1], linestyle = "--")  
    #plt.axvline(x=ilist_firstnocpl2[-1], linestyle = "--")  
    #plt.axvline(x=ilist_firstnocpl3[-1], linestyle = "--")    
else:
    plt.axvline(x=ilist[-2], color='grey') 
    plt.axvline(x=ilist_nocpl[-2], linestyle = "--", color='grey')   

#plt.title("Varying c-values for cusps 468, 487 and 505, selected network upper right (0.01*60 rate)", fontsize=10)
#plt.legend((line1, line2), ('Coupling', 'No coupling'))
#plt.title("Spatial variance approaching year {} drought scenario for region {}".format(year, region), fontsize=10)
#plt.title("Region NWS", fontsize=15)
plt.tight_layout()
#plt.gca().add_artist(lengend1)

#if no_cpl_dummy == True:
#    fig.savefig("no_coupling/spat_var_unstable_amaz_{}_{}_adaptsample{}_adaptsigma{}_{}_{}_noise{}_std1.png".format(resolution_type, 
#        year_type, str(start_file).zfill(3), int(np.around(100*adapt)), id, float(rain_fact), noise), dpi=200)
#else:

fig.savefig("spatvar_unstable_amaz_{}_final_region{}_v2.png".format(year, region), dpi=200)