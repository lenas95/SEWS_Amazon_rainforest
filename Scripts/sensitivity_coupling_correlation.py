

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
os.chdir("/p/projects/dominoes/lena/jobs/results/noise/final/{}/coupling_states".format(year))

c_lis = []

#Empyty lists to append var_items and cor_items    

kendall_tau_cpl = []
p_cpl = []
kendall_tau = []
p = []
coupling = np.arange(1.0, 5.5, 0.5)

# The scripts below calculate the Moran's I coefficient for the different regions and save the respective Kendall$\tau$ value denoting the strength of the trend
# Two different approaches of computing the Moran's I coefficient were implemented. Both giving the same result but the first one being faster
'''
regions_list = [0,1,2,3]
#Try out with another sum approach
def correlation():
    
    for region in regions_list:
        kendall_tau_cpl = []
        p_cpl = []
        for item in coupling:
            all_states = np.loadtxt("region{}/states_cusps_{}_region{}_coupling{}.txt".format(region, year, region, item), dtype = np.float64)
            if region == 0:
                neighbour = np.loadtxt("/p/projects/dominoes/lena/jobs/results/noise/neighbourslist_NWS.txt", dtype=int)
                cells = np.loadtxt("/p/projects/dominoes/lena/jobs/results/noise/NWS_cells.txt", dtype=int)
            elif region == 1:
                neighbour = np.loadtxt("/p/projects/dominoes/lena/jobs/results/noise/neighbourslist_NSA.txt", dtype=int)
                cells = np.loadtxt("/p/projects/dominoes/lena/jobs/results/noise/NSA_cells.txt", dtype=int)
            elif region == 2:
                neighbour = np.loadtxt("/p/projects/dominoes/lena/jobs/results/noise/neighbourslist_SAM.txt", dtype=int)
                cells = np.loadtxt("/p/projects/dominoes/lena/jobs/results/noise/SAM_cells.txt", dtype=int)
            else:
                neighbour = np.loadtxt("/p/projects/dominoes/lena/jobs/results/noise/neighbourslist_NES.txt", dtype=int)
                cells = np.loadtxt("/p/projects/dominoes/lena/jobs/results/noise/NES_cells.txt", dtype=int)
            print(f"Item is", item)
            sum = 0
            den_list = []
            cor =  []
            num_cols_cpl = all_states.shape[1]
            for iteration in range(0, num_cols_cpl):
                print(f"Iteration is ", iteration)
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

            #kendall_tau_only = kendalltau(cor, noise)  #The mk.original_test gives out same tau and p-values as the kendalltau(cor, noise) or kendalltau(cor, iterations)
            #print(f"kendall tau only is", kendall_tau_only)
            mk_cpl = mk.original_test(cor)
            print(f"MK coupling is", mk_cpl)
            kend_item_cpl, pval_cpl = mk_cpl.Tau, mk_cpl.p
            kendall_tau_cpl.append(kend_item_cpl)
            print(f"Shape of kendall_tau_no_cpl is_", np.shape(kendall_tau_cpl))
            p_cpl.append(pval_cpl)

        path =  '/p/projects/dominoes/lena/jobs/results/noise/final/{}/coupling_states'.format(year, region)
        fmt = '%1.2f'
        np.savetxt(os.path.join(path, "kendalltau_{}_region{}_coupling_correlation.txt".format(year, region)), kendall_tau_cpl, fmt = fmt)

    return kendall_tau_cpl, p_cpl

def correlation_nocpl():
    
    for item in noise:
        all_states_nocpl = np.loadtxt("no_coupling/region{}/states_cusps_{}_region{}_noise{}.txt".format(region, year, region, item), dtype = np.float64)
        #print(f"Iteration is", iteration)
        sum = 0
        den_list = []
        cor_nocpl =  []

        num_cols = all_states_nocpl.shape[1]
        for iteration in range(0, num_cols):
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

        mk_nocpl = mk.original_test(cor_nocpl)   #The mk.original_test gives out same tau and p-values as the kendalltau(cor, noise) or kendalltau(cor, iterations)
        print(f"Mk no coupling is", mk_nocpl)
        kend_item_nocpl, pval_nocpl = mk_nocpl.Tau, mk_nocpl.p
        kendall_tau.append(kend_item_nocpl)
        print(f"Shape of kendall_tau_no_cpl is_", np.shape(kendall_tau))
        p.append(pval_nocpl)

    return kendall_tau, p
'''

regions_list = [0,1,2,3]
#Deactivate to calculate sensitivity to coupling
#Calculate correlation from loaded states according to Dakos 2010
'''
def correlation_nocpl():

    for region in regions_list:
        kendall_tau = []
        p = []
        for element in coupling:
            all_states_nopl = np.loadtxt("no_coupling/region{}/states_cusps_{}_region{}_coupling{}.txt".format(region, year, region, element), dtype = np.float64)
            all_states = all_states[:, :-1]
            if region == 0:
                neighbour = np.loadtxt("/p/projects/dominoes/lena/jobs/results/noise/neighbourslist_NWS.txt", dtype=int)
                cells = np.loadtxt("/p/projects/dominoes/lena/jobs/results/noise/NWS_cells.txt", dtype=int)
            elif region == 1:
                neighbour = np.loadtxt("/p/projects/dominoes/lena/jobs/results/noise/neighbourslist_NSA.txt", dtype=int)
                cells = np.loadtxt("/p/projects/dominoes/lena/jobs/results/noise/NSA_cells.txt", dtype=int)
            elif region == 2:
                neighbour = np.loadtxt("/p/projects/dominoes/lena/jobs/results/noise/neighbourslist_SAM.txt", dtype=int)
                cells = np.loadtxt("/p/projects/dominoes/lena/jobs/results/noise/SAM_cells.txt", dtype=int)
            else:
                neighbour = np.loadtxt("/p/projects/dominoes/lena/jobs/results/noise/neighbourslist_NES.txt", dtype=int)
                cells = np.loadtxt("/p/projects/dominoes/lena/jobs/results/noise/NES_cells.txt", dtype=int)
            
            cor_nocpl = []

            num_cols_nocpl = all_states_nocpl.shape[1]
            j_list = []
            for j in range(0, num_cols_nocpl-1):
                num_list = []
                den_list = []
                j_list.append(j)
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


            path =  '/p/projects/dominoes/lena/jobs/results/noise/final/{}/coupling_states'.format(year, region)
            fmt = '%1.2f'
            np.savetxt(os.path.join(path, "no_coupling/correlation{}_region{}_coupling{}.txt".format(year, region, element)), (cor_nocpl, j_list), fmt = fmt)

            
            mk_nocpl = mk.original_test(cor_nocpl)  #The mk.original_test gives out same tau and p-values as the kendalltau(cor, noise) or kendalltau(cor, iterations)
            print(f"Mk no coupling is", mk_nocpl)
            kend_item_nocpl, pval_nocpl = mk_nocpl.Tau, mk_nocpl.p
            kendall_tau.append(kend_item_nocpl)
            print(f"Shape of kendall_tau_no_cpl is_", np.shape(kendall_tau))
            p.append(pval_nocpl)

    return kendall_tau, p


def correlation():

    for region in regions_list:
        kendall_tau_cpl = []
        p_cpl = []
        for element in coupling:
            all_states = np.loadtxt("region{}/states_cusps_{}_region{}_coupling{}.txt".format(region, year, region, element), dtype = np.float64)
            all_states = all_states[:, :-1]
            if region == 0:
                neighbour = np.loadtxt("/p/projects/dominoes/lena/jobs/results/noise/neighbourslist_NWS.txt", dtype=int)
                cells = np.loadtxt("/p/projects/dominoes/lena/jobs/results/noise/NWS_cells.txt", dtype=int)
            elif region == 1:
                neighbour = np.loadtxt("/p/projects/dominoes/lena/jobs/results/noise/neighbourslist_NSA.txt", dtype=int)
                cells = np.loadtxt("/p/projects/dominoes/lena/jobs/results/noise/NSA_cells.txt", dtype=int)
            elif region == 2:
                neighbour = np.loadtxt("/p/projects/dominoes/lena/jobs/results/noise/neighbourslist_SAM.txt", dtype=int)
                cells = np.loadtxt("/p/projects/dominoes/lena/jobs/results/noise/SAM_cells.txt", dtype=int)
            else:
                neighbour = np.loadtxt("/p/projects/dominoes/lena/jobs/results/noise/neighbourslist_NES.txt", dtype=int)
                cells = np.loadtxt("/p/projects/dominoes/lena/jobs/results/noise/NES_cells.txt", dtype=int)
            
            print(f"Shape of all states is", np.shape(all_states))
            print(f"Last column of all states is", all_states[:, -1])
            cor = []
            num_cols = all_states.shape[1]
            i_list = []
            for i in range(0, num_cols):
                den_list = []
                num_list = []
                i_list.append(i)
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
            path =  '/p/projects/dominoes/lena/jobs/results/noise/final/{}/coupling_states'.format(year, region)
            fmt = '%1.5f'
            np.savetxt(os.path.join(path, "region{}/correlation{}_region{}_coupling{}.txt".format(region, year, region, element)), (cor), fmt = fmt)

            #kendall_tau_only = kendalltau(cor, noise)
            #print(f"kendall tau only is", kendall_tau_only)  #The mk.original_test gives out same tau and p-values as the kendalltau(cor, noise) or kendalltau(cor, iterations)
            mk_cpl = mk.original_test(cor)
            print(f"MK coupling is", mk_cpl)
            kend_item_cpl, pval_cpl = mk_cpl.Tau, mk_cpl.p
            kendall_tau_cpl.append(kend_item_cpl)
            print(f"Shape of kendall_tau_no_cpl is_", np.shape(kendall_tau_cpl))
            p_cpl.append(pval_cpl)
        
        path =  '/p/projects/dominoes/lena/jobs/results/noise/final/{}/coupling_states'.format(year, region)
        fmt = '%1.2f'
        np.savetxt(os.path.join(path, "kendalltau_{}_region{}_coupling_correlation.txt".format(year, region)), (kendall_tau_cpl, p_cpl), fmt = fmt)

    return kendall_tau_cpl, p_cpl


kendall_cpl, pcpl = correlation()
#kendall_nocpl, pnocpl = correlation_nocpl()
'''

#all_states = np.loadtxt("region{}/states_cusps_{}_region{}_coupling{}.txt".format(region, year, region, item), dtype = np.float64)
#all_states_nocpl = np.loadtxt("no_coupling/region{}/states_cusps_{}_region{}.txt".format(region, year, region), dtype = np.float64)

#Load kendall taus for different regions
region0_cpl = np.loadtxt("kendalltau_{}_region0_coupling_correlation.txt".format(year), max_rows =1 , dtype = np.float64)
region1_cpl = np.loadtxt("kendalltau_{}_region1_coupling_correlation.txt".format(year), max_rows =1, dtype = np.float64)
region2_cpl = np.loadtxt("kendalltau_{}_region2_coupling_correlation.txt".format(year), max_rows =1 , dtype = np.float64)
region3_cpl = np.loadtxt("kendalltau_{}_region3_coupling_correlation.txt".format(year), max_rows =1 , dtype = np.float64)

#Plotting squences for variance (Final)

fig = plt.figure(figsize = (8,6))
ax1 = fig.add_subplot(111)
ax1.set_ylim(-1.0,1.0)

#coupling = [0, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600]
line1, =  ax1.plot(coupling, region0_cpl[:-2],  linestyle = "-") 
line2, = ax1.plot(coupling, region1_cpl[:-2], linestyle = "--") 
line3, = ax1.plot(coupling, region2_cpl[:-2], linestyle = "dotted")
line4, = ax1.plot(coupling, region3_cpl[:-2], linestyle = "dashdot")

#ax1.set_xlabel('Increase of the coupling strength factor')
#ax1.set_ylabel('Kendall{}'.format(r'$\mathrm{\tau}$'), color = 'black', fontsize=25)
ax1.tick_params(axis='x', labelsize = 25)
ax1.tick_params(axis='y', labelsize = 25)

plt.legend((line1, line2, line3, line4), ('Region NWS', 'Region NSA', 'Region SAM', 'Region NES'), fontsize=25)

axes = plt.gca()
#axes.ticklabel_format(style='plain', axis='y', scilimits=(0,0))
ax1.xaxis.label.set_size(17)
ax1.yaxis.label.set_size(17)
ax1.margins(x=0)

#To get the ticks on both axes
ax2 = ax1.twinx()
plt.tick_params(right=False, bottom=True)#ax2.set_ylim(0, 0.01)
#ax1.set_xticks([0,0.005,0.01,0.015,0.02,0.025,0.0]) 
#ax1.set_ylim(0, 0.01)
#ax1.axes.yaxis.set_ticklabels([]) #Disbale for first image
ax2.axes.yaxis.set_ticklabels([])
#ax2.axes.xaxis.set_ticklabels([])
ax3 = ax1.twiny()
ax3.axes.xaxis.set_ticklabels([])
ax3.tick_params(top=False)

ax1.text(0.95, 0.05,'a)',
    horizontalalignment='center',
    verticalalignment='center',
    transform = ax1.transAxes,
    fontsize=25,
    weight='bold')

plt.grid(False)
ax1.grid(False)
ax2.grid(False)

plt.tight_layout()

fig.savefig("coupling_sensitivity_{}_correlation_final.png".format(year), dpi=200)
