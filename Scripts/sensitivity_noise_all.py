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

year = 2005
no_cpl_dummy = 0
variance = 0
region = 3

if no_cpl_dummy == 0: #no_cpl_dummy can be used to shut on or off the coupling; If True then there is no coupling, False with normal coupling
    no_cpl_dummy = True 
elif no_cpl_dummy == 1:
    no_cpl_dummy = False 
else:
    print("Uncorrect value given for no_cpl_dummy, namely: {}".format(no_cpl_dummy))
    exit(1)

#Load noise states for different noise intensities

#print(f"The shape of chr_list ist", np.shape(chr_list))
#print(f"First entry fo chrlist is", chr_list[0])

#Plotting squences for variance (Final)

x = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]

data_cpl = []
data_nocpl = []

if variance == 0:
    for i in x:
        data_cpl.append(np.loadtxt('/p/projects/dominoes/lena/jobs/results/noise/final/{}/noise_states/region{}/correlation_values/correlation{}_region{}_coupling{}.txt'.format(year, region, year, region, i)))
        data_nocpl.append(np.loadtxt('/p/projects/dominoes/lena/jobs/results/noise/final/{}/noise_states/no_coupling/region{}/correlation_values/correlation{}_region{}_coupling{}.txt'.format(year, region, year, region, i)))
else:
    for i in x:
        data_cpl.append(np.loadtxt('/p/projects/dominoes/lena/jobs/results/noise/final/{}/noise_states/region{}/variance_values/variance{}_region{}_coupling{}.txt'.format(year, region, year, region, i)))
        data_nocpl.append(np.loadtxt('/p/projects/dominoes/lena/jobs/results/noise/final/{}/noise_states/no_coupling/region{}/variance_values/variance{}_region{}_coupling{}.txt'.format(year, region, year, region, i)))


#print(file1)  # Example usage

os.chdir("/p/projects/dominoes/lena/jobs/results/noise/final/{}/noise_states/region{}".format(year, region))


for item_cpl, item_nocpl, x in zip(data_cpl, data_nocpl, x):

    fig = plt.figure(figsize = (8,6))
    ax1 = fig.add_subplot(111)
    if variance == 0:
        ax1.set_ylim(-1.0,1.0)
    else:
        ax1.set_ylim(0, 0.010)

    line1, =  ax1.plot(range(0, np.shape(item_cpl)[0]), item_cpl,  linestyle = "-", color='black') 
    line2, = ax1.plot(range(0, np.shape(item_nocpl)[0]), item_nocpl, linestyle = "--", color = 'black') 

    #ax1.set_xlabel('Increase of the coupling strength factor')
    #ax1.set_ylabel('Kendall {} rank correlation'.format(r'$\mathrm{\tau}$'), color = 'black')
    ax1.tick_params(axis='x', labelsize = 17)
    ax1.tick_params(axis='y', labelsize = 17)

    plt.legend((line1, line2), ('Coupling', 'No coupling'))

    axes = plt.gca()
    #axes.ticklabel_format(style='plain', axis='y', scilimits=(0,0))
    ax1.xaxis.label.set_size(17)
    ax1.yaxis.label.set_size(17)
    ax1.margins(x=0)

    
    #To get the ticks on both axes
    ax2 = ax1.twinx()
    ax2.tick_params(right=False, bottom=True, top=False)
    #ax2.set_ylim(0, 0.01)
    #ax1.set_xticks([0,0.005,0.01,0.015,0.02,0.025,0.0]) 
    #ax1.set_ylim(0, 0.01)
    #ax1.axes.yaxis.set_ticklabels([]) #Disbale for first image
    ax2.axes.yaxis.set_ticklabels([])
    #ax2.axes.xaxis.set_ticklabels([])
    ax3 = ax1.twiny()
    ax3.axes.xaxis.set_ticklabels([])
    #ax3.axes.yaxis.set_ticklabels([])
    ax3.tick_params(top=False)
    

    if variance == 0:
        plt.title("Correlation plot for noise {}".format(x), fontsize=17)
    else:
        plt.title("Variance plot for noise {}".format(x), fontsize=17)

    plt.grid(False)
    ax1.grid(False)
    ax2.grid(False)
    ax3.grid(False)

    #plt.tight_layout()
    if variance == 0:
        fig.savefig("correlation_values/correlation_{}_region{}_noise{}.png".format(year, region, x), dpi=200)
    else:
        fig.savefig("variance_values/variance_{}_region{}_noise{}.png".format(year, region, x), dpi=200)
    #plt.clf()
    ax1.cla()
    plt.cla()
    #ax2.cla()
    #ax3.cla()
    #ax1.cla()
    #plt.cla()
    #ax2.cla()
    #ax3.cla()

    #plt.close()
    print(f"Figure saved")

'''

for item in range(10):
    for i in x:
        line1, =  ax1.plot(chr_list[item],  linestyle = "-") 
        line2, = ax1.plot(chr_list_nocpl[item], linestyle = "--") 
        if variance == 0:
            plt.savefig(os.path.join(path, "noise_states/region{}/region{}_correlation_noise{}.png".format(region, region, i)), dpi=200)
        else:
            plt.savefig(os.path.join(path, "noise_states/region{}/region{}_variance_noise{}.png".format(rregion, egion, i)), dpi=200)
        print(f"Figure saved")
        plt.cla()

file_number = 1  # A counter to keep track of the number of files you have found
path = "/p/projects/dominoes/lena/jobs/results/noise/final/2005"


if variance == 0:
    directory_nocpl = os.path.join(path, "noise_states/no_coupling/region{}/correlation_values".format(region))
    directory_cpl = os.path.join(path, "noise_states/region{}/correlation_values".format(region))

else:
    directory_nocpl = os.path.join(path, "noise_states/no_coupling/region{}/variance_values".format(region))
    directory_cpl = os.path.join(path, "noise_states/region{}/variance_values".format(region))


print(f"Directory is", directory_cpl)

noise = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]
chr_list = []
chr_list_nocpl = []

for i in noise:
    if variance == 0:
        datafile_cpl = directory_cpl+ "/correlation{}_region{}_coupling{}".format(year, region, float(i)) + ".txt"
        datafile_nocpl = directory_nocpl+ "/correlation{}_region{}_coupling{}".format(year, region, float(i)) + ".txt"
    else:
        datafile_cpl = directory_cpl+ "/variance{}_region{}_coupling{}".format(year, region, float(i)) +".txt"
        datafile_nocpl = directory_nocpl+ "/variance{}_region{}_coupling{}".format(year, region, float(i)) + ".txt"

    
    chr_list.append(open(datafile_cpl).read())
    chr_list_nocpl.append(open(datafile_nocpl).read())
'''