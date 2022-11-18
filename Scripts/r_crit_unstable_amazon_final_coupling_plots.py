

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

"This code is used to plot the evolution of the Moran's I coefficient and the spatial variance for different coupling strenghts"

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
print(np.count_nonzero((0 < c_end) & (c_end < np.sqrt(4/27))))
print(f"Percentage:", (np.count_nonzero((0 < c_end) & (c_end < np.sqrt(4/27)))/len(cells)) * 100)
print(f"Standard deviation", np.std(c_end))

t_step = 0.1
realtime_break = 100 #originally 30000 and works with 200 (see r_crt_unstable_amazon.py)
timesteps = (realtime_break/t_step)
dc = (c_end/timesteps)

os.chdir("/p/projects/dominoes/lena/jobs/results/noise/final/{}/coupling_states".format(year))

c_lis = []

coupling = np.arange(1.0, 6.5, 0.5)


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

#Try out with another sum approach
for iteration in coupling:
    cor =  []

    all_states = np.loadtxt("region{}/states_cusps_{}_region{}_coupling{}.txt".format(region, year, region, iteration), dtype = np.float64)
    # print(f"Iteration is", iteration)
    num_cols = all_states.shape[1]
    for column_cpl in range(0, num_cols):
        sum = 0
        den_list = []
        avg = np.mean(all_states[:, column_cpl])
        for i in range(0, len(list(cells))):
            for j in range(0, len(list(cells))):
                cell_pair = [cells[i], cells[j]]
                if (cell_pair == neighbour).all(1).any() == True:
                    sum = sum + ((all_states[i, column_cpl] - avg)*(all_states[j, column_cpl] - avg))
                    #print(cells[i], cells[j])
                    #print(f"Sum is", sum)
        
        #Calculate denominator for 2-point-correlation
        for x in range(0, len(list(cells))):
            den_item = np.square(all_states[x, column_cpl] - avg)
            den_list.append(den_item)

        #Calculate correlation for every iteration
        corr_item = len(list(cells)) * (sum) /(len(list(neighbour)) * np.sum(den_list))
        cor.append(corr_item)

    cor_nocpl =  []
    all_states_nocpl = np.loadtxt("no_coupling/region{}/states_cusps_{}_region{}.txt".format(region, year, region), dtype = np.float64)
    num_cols_nocpl = all_states_nocpl.shape[1]

    for column in range(0, num_cols_nocpl):
        #print(f"Iteration is", iteration)
        sum = 0
        den_list = []
        avg = np.mean(all_states_nocpl[:, column])
        for i in range(0, len(list(cells))):
            for j in range(0, len(list(cells))):
                cell_pair = [cells[i], cells[j]]
                if (cell_pair == neighbour).all(1).any() == True:
                    sum = sum + ((all_states_nocpl[i, column] - avg)*(all_states_nocpl[j, column] - avg))
                    #print(cells[i], cells[j])
                    #print(f"Sum is", sum)
        
        #Calculate denominator for 2-point-correlation
        for x in range(0, len(list(cells))):
            den_item = np.square(all_states_nocpl[x, column] - avg)
            den_list.append(den_item)

        #Calculate correlation for every iteration
        corr_item = len(list(cells)) * (sum) /(len(list(neighbour)) * np.sum(den_list))
        cor_nocpl.append(corr_item)

    # Print out tau kendall value for the whole time series
    mk_cpl = mk.original_test(cor)
    mk_nocpl = mk.original_test(cor_nocpl)
    print(f"Mk is", mk_cpl)
    print(f"Mk no coupling is", mk_nocpl)
    kend_item_cpl = round(mk_cpl.Tau,2)
    kend_item_nocpl = round(mk_nocpl.Tau,2)
    kend_items = [kend_item_cpl, kend_item_nocpl]
    #kend_item_cpl_only = kendalltau(cor, ilist)
    #kend_item_nocpl_only = kendalltau(cor_no_cpl, ilist_nocpl)
    #print(f"Kendalltauitem coupling", kend_item_cpl_only)
    #print(f"Kendalltau item no coupling", kend_item_nocpl_only)
    '''

#coupling = np.arange(4.0, 6.5, 0.5)
#Try out with another sum approach
regions = [0,1,2,3]
corr_dummy = 1

if corr_dummy == 1:
    corr_dummy = True
else:
    corr_dummy = False

if corr_dummy == True:
    for region in regions:
        
        '''
        cor_list_0 = np.loadtxt("region{}/correlation{}_region{}_coupling1.0.txt".format(region, year, region), dtype=np.float64)
        cor_list_4 = np.loadtxt("region{}/correlation{}_region{}_coupling4.0.txt".format(region, year, region), dtype=np.float64)
        cor_list_45 = np.loadtxt("region{}/correlation{}_region{}_coupling4.5.txt".format(region, year, region), dtype=np.float64)
        cor_list_5 = np.loadtxt("region{}/correlation{}_region{}_coupling5.0.txt".format(region, year, region), dtype=np.float64)
        cor_list_55 = np.loadtxt("region{}/correlation{}_region{}_coupling5.5.txt".format(region, year, region), dtype=np.float64)
        cor_list_6 = np.loadtxt("region{}/correlation{}_region{}_coupling6.0.txt".format(region, year, region), dtype=np.float64)
        #Plotting squences for variance and correlation

        fig = plt.figure(figsize = (8,6))
        ax1 = fig.add_subplot(111)


        line1, =  ax1.plot(range(0, len(cor_list_0)), cor_list_0) # coupling
        line2, =  ax1.plot(range(0, len(cor_list_4)), cor_list_4) # coupling
        line3, =  ax1.plot(range(0, len(cor_list_45)), cor_list_45) # coupling
        line4, =  ax1.plot(range(0, len(cor_list_5)), cor_list_5) # coupling
        line5, =  ax1.plot(range(0, len(cor_list_55)), cor_list_55) # coupling
        line6, =  ax1.plot(range(0, len(cor_list_6)), cor_list_6) # coupling
        '''

        cor_list_0 = np.loadtxt("region{}/correlation{}_region{}_coupling1.0.txt".format(region, year, region), dtype=np.float64)
        cor_list_15 = np.loadtxt("region{}/correlation{}_region{}_coupling1.5.txt".format(region, year, region), dtype=np.float64)
        cor_list_2 = np.loadtxt("region{}/correlation{}_region{}_coupling2.0.txt".format(region, year, region), dtype=np.float64)
        cor_list_25 = np.loadtxt("region{}/correlation{}_region{}_coupling2.5.txt".format(region, year, region), dtype=np.float64)
        cor_list_3 = np.loadtxt("region{}/correlation{}_region{}_coupling3.0.txt".format(region, year, region), dtype=np.float64)
        cor_list_35 = np.loadtxt("region{}/correlation{}_region{}_coupling3.5.txt".format(region, year, region), dtype=np.float64)
        cor_list_4 = np.loadtxt("region{}/correlation{}_region{}_coupling4.0.txt".format(region, year, region), dtype=np.float64)
        cor_list_45 = np.loadtxt("region{}/correlation{}_region{}_coupling4.5.txt".format(region, year, region), dtype=np.float64)
        cor_list_5 = np.loadtxt("region{}/correlation{}_region{}_coupling5.0.txt".format(region, year, region), dtype=np.float64)
        #Plotting squences for variance and correlation

        fig = plt.figure(figsize = (8,6))
        ax1 = fig.add_subplot(111)

        '''
        line1, =  ax1.plot(range(0, len(cor_list_0)), cor_list_0) # coupling
        line2, =  ax1.plot(range(0, len(cor_list_4)), cor_list_4) # coupling
        line3, =  ax1.plot(range(0, len(cor_list_45)), cor_list_45) # coupling
        line4, =  ax1.plot(range(0, len(cor_list_5)), cor_list_5) # coupling
        line5, =  ax1.plot(range(0, len(cor_list_55)), cor_list_55) # coupling
        line6, =  ax1.plot(range(0, len(cor_list_6)), cor_list_6) # coupling
        '''

        line1, =  ax1.plot(range(0, len(cor_list_0)), cor_list_0) # coupling
        line2, =  ax1.plot(range(0, len(cor_list_15)), cor_list_15) # coupling
        line3, =  ax1.plot(range(0, len(cor_list_2)), cor_list_2) # coupling
        line4, =  ax1.plot(range(0, len(cor_list_25)), cor_list_25) # coupling
        line5, =  ax1.plot(range(0, len(cor_list_3)), cor_list_3) # coupling
        line6, =  ax1.plot(range(0, len(cor_list_35)), cor_list_35) # coupling
        line7, =  ax1.plot(range(0, len(cor_list_4)), cor_list_4) # coupling
        line8, =  ax1.plot(range(0, len(cor_list_45)), cor_list_45) # coupling
        line9, =  ax1.plot(range(0, len(cor_list_5)), cor_list_5) # coupling

        ax2 = ax1.twinx()
        plt.tick_params(right=False, bottom=True)
        #ax2.set_ylim(0, 0.01)
        # ax1.set_yticks([0,0.001,0.002,0.003,0.004,0.005,0.006,0.007,0.008,0.009,0.01]) 
        ax1.set_ylim(-1.0, 1.0)
        if region == 0: 
            pass
        else:
            ax1.axes.yaxis.set_ticklabels([])        
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
        '''
        textstr = ', '.join((
            r'$\mathrm{Kendall \tau}=%.2f$' % (kend_item_cpl, ),
            r'$\mathrm{p}<%.2f$' % (0.05, )))

        textstr_nocpl = ', '.join((
            r'$\mathrm{Kendall \tau}=%.2f$' % (kend_item_nocpl, ),
            r'$\mathrm{p}<%.2f$' % (0.05, )))
        '''

        #ax1.text(0.03, 0.6, textstr, transform=ax1.transAxes, fontsize=23, color='royalblue')
        #ax1.text(0.03, 0.53, textstr_nocpl, transform=ax1.transAxes, fontsize=23, color='maroon')
       
        if region == 0:
            letter = 'e)'
        elif region == 1:
            letter = 'f)'
        elif region == 2:
            letter  = 'g)'
        else:
            letter = 'h)'

        ax1.text(0.95, 0.95,'{}'.format(letter),
            horizontalalignment='center',
            verticalalignment='center',
            transform = ax1.transAxes,
            fontsize=20,
            weight='bold')
        

        #plt.legend((line1,line2, line3, line4, line5, line6), ('Original','4x', '4.5x','5x','5.5x', '6x'), loc='upper right', fontsize=20)
        # plt.legend((line1,line2, line3, line4, line5, line6, line7, line8, line9), ('Original','1.5x', '2x','2.5x','3x','3.5x', '4x', '4.5x', '5x'), loc='lower right', fontsize=12)
        #plt.title("Spatial variance approaching year {} drought scenario for region {}".format(year, region), fontsize=10)
        #ax1.axvline(x=ilist[-2], color='grey') 
        #ax1.axvline(x=ilist_nocpl[-2], linestyle = "--", color='grey') 
        plt.tight_layout()
        #fig.savefig("plots/all_correlation_unstable_amaz_{}_final_region{}.png".format(year, region), dpi=200)
        fig.savefig("plots/all_1xto5x_correlation_unstable_amaz_{}_final_region{}.png".format(year, region), dpi=200)
else:

    for region in regions:
        '''
        list_0 = np.loadtxt("region{}/variance{}_region{}_coupling1.0.txt".format(region, year, region), dtype=np.float64)
        list_4 = np.loadtxt("region{}/variance{}_region{}_coupling4.0.txt".format(region, year, region), dtype=np.float64)
        list_45 = np.loadtxt("region{}/variance{}_region{}_coupling4.5.txt".format(region, year, region), dtype=np.float64)
        list_5 = np.loadtxt("region{}/variance{}_region{}_coupling5.0.txt".format(region, year, region), dtype=np.float64)
        list_55 = np.loadtxt("region{}/variance{}_region{}_coupling5.5.txt".format(region, year, region), dtype=np.float64)
        list_6 = np.loadtxt("region{}/variance{}_region{}_coupling6.0.txt".format(region, year, region), dtype=np.float64)
        #Plotting squences for variance and correlation

        fig = plt.figure(figsize = (8,6))
        ax1 = fig.add_subplot(111)


        line1, =  ax1.plot(range(0, len(list_0)), list_0) # coupling
        line2, =  ax1.plot(range(0, len(list_4)), list_4) # coupling
        line3, =  ax1.plot(range(0, len(list_45)), list_45) # coupling
        line4, =  ax1.plot(range(0, len(list_5)), list_5) # coupling
        line5, =  ax1.plot(range(0, len(list_55)), list_55) # coupling
        line6, =  ax1.plot(range(0, len(list_6)), list_6) # coupling
        '''

        list_0 = np.loadtxt("region{}/variance{}_region{}_coupling1.0.txt".format(region, year, region), dtype=np.float64)
        list_15 = np.loadtxt("region{}/variance{}_region{}_coupling1.5.txt".format(region, year, region), dtype=np.float64)
        list_2 = np.loadtxt("region{}/variance{}_region{}_coupling2.0.txt".format(region, year, region), dtype=np.float64)
        list_25 = np.loadtxt("region{}/variance{}_region{}_coupling2.5.txt".format(region, year, region), dtype=np.float64)
        list_3 = np.loadtxt("region{}/variance{}_region{}_coupling3.0.txt".format(region, year, region), dtype=np.float64)
        list_35 = np.loadtxt("region{}/variance{}_region{}_coupling3.5.txt".format(region, year, region), dtype=np.float64)
        list_4 = np.loadtxt("region{}/variance{}_region{}_coupling4.0.txt".format(region, year, region), dtype=np.float64)
        list_45 = np.loadtxt("region{}/variance{}_region{}_coupling4.5.txt".format(region, year, region), dtype=np.float64)
        list_5 = np.loadtxt("region{}/variance{}_region{}_coupling5.0.txt".format(region, year, region), dtype=np.float64)
        #Plotting squences for variance and correlation

        fig = plt.figure(figsize = (8,6))
        ax1 = fig.add_subplot(111)


        line1, =  ax1.plot(range(0, len(list_0)), list_0) # coupling
        line2, =  ax1.plot(range(0, len(list_15)), list_15) # coupling
        line3, =  ax1.plot(range(0, len(list_2)), list_2) # coupling
        line4, =  ax1.plot(range(0, len(list_25)), list_25) # coupling
        line5, =  ax1.plot(range(0, len(list_3)), list_3) # coupling
        line6, =  ax1.plot(range(0, len(list_35)), list_35) # coupling
        line7, =  ax1.plot(range(0, len(list_4)), list_4) # coupling
        line8, =  ax1.plot(range(0, len(list_45)), list_45) # coupling
        line9, =  ax1.plot(range(0, len(list_5)), list_5) # coupling

        ax2 = ax1.twinx()
        plt.tick_params(right=False, bottom=True)
        #ax2.set_ylim(0, 0.01)
        #ax1.set_yticks([0,0.001,0.002,0.003,0.004,0.005,0.006,0.007,0.008,0.009,0.01]) 
        ax1.set_yticks([0,0.002,0.004,0.006,0.008,0.01]) 
        # ax1.set_ylim(0, 1.0)

        if region == 0: 
            pass
        else:
            ax1.axes.yaxis.set_ticklabels([]) 

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
        '''
        textstr = ', '.join((
            r'$\mathrm{Kendall \tau}=%.2f$' % (kend_item_cpl, ),
            r'$\mathrm{p}<%.2f$' % (0.05, )))

        textstr_nocpl = ', '.join((
            r'$\mathrm{Kendall \tau}=%.2f$' % (kend_item_nocpl, ),
            r'$\mathrm{p}<%.2f$' % (0.05, )))
        '''


        '''
        #ax1.text(0.03, 0.6, textstr, transform=ax1.transAxes, fontsize=23, color='royalblue')
        #ax1.text(0.03, 0.53, textstr_nocpl, transform=ax1.transAxes, fontsize=23, color='maroon')
        '''
        if region == 0:
            letter = 'a)'
        elif region == 1:
            letter = 'b)'
        elif region == 2:
            letter  = 'c)'
        else:
            letter = 'd)'
        
        ax1.text(0.95, 0.95,'{}'.format(letter),
            horizontalalignment='center',
            verticalalignment='center',
            transform = ax1.transAxes,
            fontsize=25,
            weight='bold')
    

        #plt.legend((line1,line2, line3, line4, line5, line6), ('Original','4x', '4.5x','5x','5.5x', '6x'), loc='upper right', fontsize=20)
        if region == 0:
            plt.legend((line1,line2, line3, line4, line5, line6, line7, line8, line9), ('Original','1.5x', '2x','2.5x','3x','3.5x', '4x', '4.5x', '5x'), loc='upper left', fontsize=20)
        else:
            pass 
        #plt.title("Spatial variance approaching year {} drought scenario for region {}".format(year, region), fontsize=10)
        #ax1.axvline(x=ilist[-2], color='grey') 
        #ax1.axvline(x=ilist_nocpl[-2], linestyle = "--", color='grey') 
        plt.tight_layout()
        #fig.savefig("plots/all_variance_unstable_amaz_{}_final_region{}.png".format(year, region), dpi=200)
        fig.savefig("plots/all_1xto5x_variance_unstable_amaz_{}_final_region{}.png".format(year, region), dpi=200)





