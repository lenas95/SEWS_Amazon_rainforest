

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
year = 2005
region = 3
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

'''
#Try out correlation calculation from Donangelo
def correlation_nocpl():
    cor_nocpl = []
    for j in range(0, len(ilist_nocpl)):
        num_list = []
        den_list = []
        den_list1 = []
        for i in range(0, len(list(neighbour))):
            item_1 = (all_states_nocpl[list(cells).index(neighbour[i,0]), j])*(all_states_nocpl[list(cells).index(neighbour[i,1]), j])
            item_2 = (all_states_nocpl[list(cells).index(neighbour[i,0]), j])
            item_3 = (all_states_nocpl[list(cells).index(neighbour[i,1]), j])
            num_list.append(item_1)
            den_list.append(item_2)
            den_list1.append(item_3)
        num = np.sum(num_list) / len(list(cells))
        den = (np.sum(den_list)/len(list(cells))) * (np.sum(den_list1)/len(list(cells)))
        corr_item = num - den
        cor_nocpl.append(corr_item)
    return cor_nocpl

def correlation():
    cor = []
    for j in range(0, len(ilist)):
        num_list = []
        den_list = []
        den_list1 = []
        for i in range(0, len(list(neighbour))):
            item_1 = (all_states[list(cells).index(neighbour[i,0]), j])*(all_states[list(cells).index(neighbour[i,1]), j])
            item_2 = (all_states[list(cells).index(neighbour[i,0]), j])
            item_3 = (all_states[list(cells).index(neighbour[i,1]), j])
            num_list.append(item_1)
            den_list.append(item_2)
            den_list1.append(item_3)
        num = np.sum(num_list) / len(list(cells))
        den = (np.sum(den_list)/len(list(cells))) * (np.sum(den_list1)/len(list(cells)))
        corr_item = num - den
        cor.append(corr_item)
    return cor



#Calculate correlation from loaded states according to Dakos 2012
def correlation_nocpl():
    cor_nocpl = []
    for j in range(0, len(ilist_nocpl)):
        num_list = []
        den_list = []
        for i in range (0, len(list(neighbour))):
            #print(neighbour[i,0])
            num_1 = all_states_nocpl[list(cells).index(neighbour[i,0]), j] - np.mean(all_states_nocpl[:,j])
            #print(list(cells).index(neighbour[i,0]))
            #print(all_states_nocpl[list(cells).index(neighbour[i,0]), j])
            num_2 =  all_states_nocpl[list(cells).index(neighbour[i,1]), j] - np.mean(all_states_nocpl[:,j])
            num_item = num_1 * num_2
            num_list.append(num_item)
        num = np.sum(num_list)

        for x in range(0, len(list(cells))):
            den_item = np.square(all_states_nocpl[x, j] - np.mean(all_states_nocpl[:,j]))
            den_list.append(den_item)
        
        den = np.sum(den_list)
        corr_item = len(list(cells)) * (num) / (len(list(neighbour)) * den)
        cor_nocpl.append(corr_item)
    return cor_nocpl

def correlation():
    cor = []
    for i in range(0, len(ilist)):
        den_list = []
        num_list = []
        #Calculate numerator of 2-point-correlation
        for x in range(0, len(list(neighbour))):
            num_item = (all_states[list(cells).index(neighbour[x,0]),i] - (np.sum(all_states[:,i])/len(list(cells)))) * (all_states[list(cells).index(neighbour[x,1]),i] - (np.sum(all_states[:,i])/len(list(cells))))
            num_list.append(num_item)
            #print(f"num_list_1:", np.shape(num_list_1))
        # num_list.append(np.sum(num_list_1))
        num = np.sum(num_list)

        #Calculate denominator for 2-point-correlation
        for j in range(0, len(list(cells))):
            den_item = np.square(all_states[j, i] - (np.sum(all_states[:,i])/len(list(cells))))
            den_list.append(den_item)
        #den_list.append(np.sum(den_list_1))
        den = np.sum(den_list)
        #print(f"Shape of den_list is", np.shape(den))
        
        #Calculate correlation for every iteration
        corr_item = len(list(cells)) * (num) /(len(list(neighbour)) * den)
        cor.append(corr_item)
        #print(f"Shape of corr_item is:", np.shape(cor))
    return cor
'''

#Try out with another sum approach
def correlation():
    cor =  []
    for iteration in range(0, len(ilist)):
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
    for iteration in range(0, len(ilist_nocpl)):
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


cpl_cell = np.loadtxt("cusp_{}_region{}.txt".format(year, region), dtype=np.float64)
no_cpl_cell = np.loadtxt("no_coupling/cusp_{}_region{}.txt".format(year, region), dtype = np.float64)

# Print out tau kendall value for the whole time series
mk_cpl = mk.original_test(cor_cpl)
mk_nocpl = mk.original_test(cor_no_cpl)
print(f"Mk is", mk_cpl)
print(f"Mk no coupling is", mk_nocpl)
kend_item_cpl = round(mk_cpl.Tau,2)
kend_item_nocpl = round(mk_nocpl.Tau,2)
kend_items = [kend_item_cpl, kend_item_nocpl]
kend_item_cpl_only = kendalltau(cor_cpl, ilist)
kend_item_nocpl_only = kendalltau(cor_no_cpl, ilist_nocpl)
print(f"Kendalltauitem coupling", kend_item_cpl_only)
print(f"Kendalltau item no coupling", kend_item_nocpl_only)

'''
#Calculate p-value for Moran's coeffiecient, whereby 0-hypthesis is Moran's = 0 for perfect radnomness
zero_hypothesis_nocpl = np.zeros((len(cor_no_cpl)))
zero_hypothesis_cpl = np.zeros((len(cor_cpl)))

stand_cpl = np.std(cor_cpl)
stand_nocpl = np.std(cor_no_cpl)
z_cpl = np.mean(cor_cpl) - np.mean(zero_hypothesis_cpl) / stand_cpl
z_no_cpl = np.mean(cor_no_cpl) - np.mean(zero_hypothesis_nocpl) / stand_nocpl
print(f"Z-score for coupling is", z_cpl)
print(f"Z-Score for no coupling is", z_no_cpl)
print(f"P-value for coupling", stats.norm.cdf(z_cpl))
print(f"P-value for no coupling", stats.norm.cdf(z_no_cpl))
print(f"p-value for no_coupling", stats.ttest_ind(cor_no_cpl, zero_hypothesis_nocpl))
print(f"p-value for coupling", stats.ttest_ind(cor_cpl, zero_hypothesis_cpl))
'''


#Plotting squences for variance and correlation

fig = plt.figure(figsize = (8,6))
ax1 = fig.add_subplot(111)
ax1.set_ylim(-1.0,1.8)

#Load coupling and no_coupling values for tipped cells
line1, =  ax1.plot(ilist, cpl_cell, 'y', linestyle = "-") # coupling
line2, = ax1.plot(ilist_nocpl, no_cpl_cell, 'y', linestyle = "--") #no coupling

ax1.set_xlabel('Timeseries')
ax1.set_ylabel('Average state for tipped cells', color = 'black')
ax1.tick_params(axis='x', labelsize = 8)
ax1.tick_params(axis='y', labelsize = 8)

ax2 = ax1.twinx()
ax2.set_ylim(0, 1.0)

line3, = ax2.plot(ilist, cor_cpl, 'g')
line4, = ax2.plot(ilist_nocpl, cor_no_cpl, 'g', linestyle = "--")

#lengend1 = plt.legend(kend_items, ['Kendtall-tau', 'Kendall-tau no coupling'], loc = 'center left')

textstr = '\n'.join((
    r'$\mathrm{Kendall-tau}=%.2f$' % (kend_item_cpl, ),
    r'$\mathrm{Kendall-tau(nocpl)}=%.2f$' % (kend_item_nocpl, )))

ax1.text(0.01, 0.7, textstr, transform=ax1.transAxes, fontsize=8)
#bbox=dict(facecolor='black', alpha=0.5),

if year == 2005:
    if region == 0:
        plt.legend((line1, line2, line3, line4), 
        ('Average state for cell 47', 'Average state for cell 47 no coupling', 'Spatial correlation', 'Spatial correlation no coupling'), prop={'size': 8}, loc='upper left')
    elif region == 1:
        plt.legend((line1, line2, line3, line4), 
        ('Average state for cell 172', 'Average state for cell 172 no coupling', 'Spatial correlation', 'Spatial correlation no coupling'), prop={'size': 8}, loc='upper left')
    elif region == 2:
        plt.legend((line1, line2, line3, line4), 
        ('Average state for cell 440', 'Average state for cell 440 no coupling', 'Spatial correlation', 'Spatial correlation no coupling'), prop={'size': 8}, loc='upper left')
    elif region == 3:
        plt.legend((line1, line2, line3, line4), 
        ('Average state for cell 538', 'Average state for cell 538 no coupling', 'Spatial correlation', 'Spatial correlation no coupling'), prop={'size': 8}, loc='upper left')
    else:
        plt.legend((line1, line2, line3, line4), 
        ('Average state for cell 47', 'Average state for cell 47 no coupling', 'Spatial correlation', 'Spatial correlation no coupling'), prop={'size': 8}, loc='upper left')
elif year == 2007:
    if region == 0:
        plt.legend((line1, line2, line3, line4), 
        ('Average state for cell 30', 'Average state for cell 30 no coupling', 'Spatial correlation', 'Spatial correlation no coupling'), prop={'size': 8}, loc='upper left')
    elif region == 1:
        plt.legend((line1, line2, line3, line4), 
        ('Average state for cell 218', 'Average state for cell 218 no coupling', 'Spatial correlation', 'Spatial correlation no coupling'), prop={'size': 8}, loc='upper left')
    elif region == 2:
        plt.legend((line1, line2, line3, line4), 
        ('Average state for cell 400', 'Average state for cell 400 no coupling', 'Spatial correlation', 'Spatial correlation no coupling'), prop={'size': 8}, loc='upper left')
    elif region == 3:
        plt.legend((line1, line2, line3, line4), 
        ('Average state for cell 540', 'Average state for cell 540 no coupling', 'Spatial correlation', 'Spatial correlation no coupling'), prop={'size': 8}, loc='upper left')
    else:
        plt.legend((line1, line2, line3, line4), 
        ('Average state for cell 30', 'Average state for cell 30 no coupling', 'Spatial correlation', 'Spatial correlation no coupling'), prop={'size': 8}, loc='upper left')
else:
    if region == 0:
        plt.legend((line1, line2, line3, line4), 
        ('Average state for cell 30', 'Average state for cell 30 no coupling', 'Spatial correlation', 'Spatial correlation no coupling'), prop={'size': 8}, loc='upper left')
    elif region == 1:
        plt.legend((line1, line2, line3, line4), 
        ('Average state for cell 216', 'Average state for cell 216 no coupling', 'Spatial correlation', 'Spatial correlation no coupling'), prop={'size': 8}, loc='upper left')
    elif region == 2:
        plt.legend((line1, line2, line3, line4), 
        ('Average state for cell 440', 'Average state for cell 440 no coupling', 'Spatial correlation', 'Spatial correlation no coupling'), prop={'size': 8}, loc='upper left')
    elif region == 3:
        plt.legend((line1, line2, line3, line4), 
        ('Average state for cell 538', 'Average state for cell 538 no coupling', 'Spatial correlation', 'Spatial correlation no coupling'), prop={'size': 8}, loc='upper left')
    else:
        plt.legend((line1, line2, line3, line4), 
        ('Average state for cell 30', 'Average state for cell 30 no coupling', 'Spatial correlation', 'Spatial correlation no coupling'), prop={'size': 8}, loc='upper left')

ax2.set_ylabel('Spatial correlation', color = 'g' )
ax2.tick_params(axis='y', labelsize = 8)
axes = plt.gca()
axes.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
ax1.xaxis.label.set_size(10)
ax1.yaxis.label.set_size(10)
ax2.yaxis.label.set_size(10)


#Load ilist for first cell tipped when plotting sholw network
if region == 4:
    #ilist_first0 = np.loadtxt("i_list_{}_region0.txt".format(year), dtype = int)
    #ilist_first1 = np.loadtxt("i_list_{}_region1.txt".format(year), dtype = int)
    #ilist_first2 = np.loadtxt("i_list_{}_region2.txt".format(year), dtype = int)
    #ilist_first3 = np.loadtxt("i_list_{}_region3.txt".format(year), dtype = int)
    #ilist_firstnocpl0 = np.loadtxt("no_coupling/i_list_{}_region0.txt".format(year), dtype = int)
    #ilist_firstnocpl1 = np.loadtxt("no_coupling/i_list_{}_region1.txt".format(year), dtype = int)
    #ilist_firstnocpl2 = np.loadtxt("no_coupling/i_list_{}_region2.txt".format(year), dtype = int)
    #ilist_firstnocpl3 = np.loadtxt("no_coupling/i_list_{}_region3.txt".format(year), dtype = int)
    for i in range(0, 301):
        plt.axvline(x=i*100, color='red') 
        plt.title("Spatial correlation approaching year {} drought scenario for region {}".format(year, region), fontsize=10)
        plt.tight_layout()
        fig.savefig("movie/correlation_unstable_amaz_{}_final_region{}_pic{}.png".format(year, region, i), dpi=200)
        #plt.axvline(x=ilist_first1[-1])
        #plt.axvline(x=ilist_first2[-1])
        #plt.axvline(x=ilist_first3[-1])
        #plt.axvline(x=ilist_firstnocpl0[-1], linestyle = "--")
        #plt.axvline(x=ilist_firstnocpl1[-1], linestyle = "--")  
        #plt.axvline(x=ilist_firstnocpl2[-1], linestyle = "--")  
        #plt.axvline(x=ilist_firstnocpl3[-1], linestyle = "--")    
else:
    '''
    if region == 0:
        ax3 = ax1.twinx()
        for i in np.arange(0, ilist[-1]+1, 100):
            print(i)
            if i % 500 == 0 or i == ilist[-1]:
                ax3.tick_params(axis='y',left=False, right=False,labelleft=False, labelright=False)
                plt.axvline(x=i, color='red')
                plt.axvline(x=ilist[-1]) 
                plt.axvline(x=ilist_nocpl[-1], linestyle = "--") 
                #plt.axvline(x=ilist[-1], color='red')
                plt.title("Spatial correlation approaching year {} drought scenario for region {}".format(year, region), fontsize=10)
                plt.tight_layout()
                fig.savefig("correlation_status/region{}/cor/correlation_unstable_amaz_{}_final_region{}_pic{}.png".format(region, year, region, i), dpi=200)
                ax3.cla()
                print(f"Figure saved")
            else:
                pass
            
    else:
        ax3 = ax1.twinx()
        for i in np.arange(0, ilist[-1]+1, 100):
            if i % 1000 == 0 or i == ilist[-1]:
                print(i)
                ax3.tick_params(axis='y',left=False, right=False,labelleft=False, labelright=False)
                plt.axvline(x=i, color='red')
                plt.axvline(x=ilist[-1]) 
                plt.axvline(x=ilist_nocpl[-1], linestyle = "--") 
                plt.title("Spatial correlation approaching year {} drought scenario for region {}".format(year, region), fontsize=10)
                plt.tight_layout()
                fig.savefig("correlation_status/region{}/cor/correlation_unstable_amaz_{}_final_region{}_pic{}.png".format(region, year, region, i), dpi=200)
                ax3.cla()
                print(f"Figure saved")
    '''
    plt.axvline(x=ilist[-1]) 
    plt.axvline(x=ilist_nocpl[-1], linestyle = "--")   


#plt.title("Varying c-values for cusps 468, 487 and 505, selected network upper right (0.01*60 rate)", fontsize=10)
plt.title("Spatial correlation approaching year {} drought scenario for region {}".format(year, region), fontsize=10)
plt.tight_layout()
#plt.gca().add_artist(lengend1)

#if no_cpl_dummy == True:
#    fig.savefig("no_coupling/spat_var_unstable_amaz_{}_{}_adaptsample{}_adaptsigma{}_{}_{}_noise{}_std1.png".format(resolution_type, 
#        year_type, str(start_file).zfill(3), int(np.around(100*adapt)), id, float(rain_fact), noise), dpi=200)
#else:

fig.savefig("newaxis_correlation_unstable_amaz_{}_final_region{}_v2.png".format(year, region), dpi=200)
print(f"Figure saved")
