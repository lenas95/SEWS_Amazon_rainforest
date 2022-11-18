import time
import numpy as np
import networkx as nx
import glob
import re
import os

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

"This code can be used to calculate the correlation between neighbouring cells (and plot it against average state of cell) for specially selected cells when increasing c_value for one/or more cells"
"The spatial correlation index was defined as the two-point correlation for all pairs of cells separated by distance 1, using the Moran’s coefficient (Legendre)"

sys_var = np.array(sys.argv[2:])
year = sys_var[0]
no_cpl_dummy = int(sys_var[1])
adapt = float(sys_var[2])       #Range of adaptability in multiples of the standard deviation; higher adapt_fact means higher adaptability
start_file = sys_var[3]

#year = 2004
#no_cpl_dummy = 1
#adapt = 1.0
#start_file = 1

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

#evaluated drought (or not drought) year for the roughest periods from 2003-2014
# hydro_1 = np.sort(np.array(glob.glob("more_data/monthly_data/correct_data/both_1deg/*{}*.nc".format(int(year)-1))))[9:]
# hydro_2 = np.sort(np.array(glob.glob("more_data/monthly_data/correct_data/both_1deg/*{}*.nc".format(int(year)))))[:9]
# data_eval = np.concatenate((hydro_1, hydro_2))

#Load neighbourlist to compute correlation between cells from r_crit_unstable_amazon.py
neighbour = np.loadtxt("./jobs/results/noise/neighbourslist.txt", dtype=int)

#Tipping function
def tip_cor( net , initial_state ):
    
    ev = evolve( net , initial_state )
    tolerance = 0.01
    

    if not ev.is_equilibrium(tolerance):
        #This network is not in equilibrium since it[0] (= moisture transport) is not effectless at the current state of the network,
      print("Warning: Initial state is not a fixed point of the system")
    elif not ev.is_stable():
        print("Warning: Initial state is not a stable point of the system")

    t_step = 1
    realtime_break = 200 #originally 30000 and works with 200 (see r_crt_unstable_amazon.py)

    dc = np.array([0.01] * 60)  # with dc = np.array([0.01] * 20)  and realtime_break = 200 works for 0.85 tipping cascade and for 1.0: rain_fact use [0.05] *20

    # Create zero arrays to store data

    c_lis = []
    # cells = list(range(446, 454)) + list(range(466, 473)) + list(range(485,492)) + list(range(504, 510)) + list(range(521,525)) + list(range(534, 537)) +list(range(544, 547))
    # cells = list(range(466, 473)) + list(range(485,492)) + list(range(504, 510)) + list(range(521,525)) + list(range(534, 537))
    corr = []
    cusp505_av = []
    cusp468_av = []
    cusp487_av = []
    kendall = []
    c_cells = [468, 487, 505]
   
    
    for i in range(0, len(dc)):
        
        #Initialize arrays to fill in according to Dakos et al. 2010, correlation formula
        sq_array = np.empty((1, 0))
        num_array = np.empty((1, 0))

        #c = net.nodes[505]['data'].get_par()['c']  # When only one cell is forced uncomment
        #net.set_param(505, 'c', c+dc[i])
        for cell in c_cells:
            c = net.nodes[cell]['data'].get_par()['c'] 
            net.set_param(cell, 'c', c+dc[i])
        c_lis.append(c) # Append c to list so can be plotted on x-axis against spatial variance

        print(f"Value for c of cell 505:", c)
        ev = evolve (net, ev.get_timeseries()[1][-1])   # making new state as dynamical system for every increased c value
        
        #sigmas = np.random.uniform(low=0, high=0.01, size=net.number_of_nodes())
        #sigmas[sigmas < 0] = 0.0
        sigmas = np.array([0.01] * np.ones(net.number_of_nodes()))
        alphas = 2 * np.ones(net.number_of_nodes())
        ev.integrate(t_step, realtime_break, sigmas = sigmas, noise = "normal", alphas = alphas, seed = None)

        #Get c-value at which the first cell tipps
        print(f"Value of c for first tipped is:", c if net.get_tip_states(ev.get_timeseries()[1][-1])[:].any() != False else "None")

        #Get the average of last state for all cusps 
        cusp_all = ev.get_timeseries()[1][-1]
        cusp_all = np.mean(cusp_all)

        for x in range(0, len(list(neighbour))):
            
            #Get all the values for one c for the numerator
            num_item = (ev.get_timeseries()[1][-1].T[neighbour[x,0]] - cusp_all) * (ev.get_timeseries()[1][-1].T[neighbour[x,1]] - cusp_all)
            num_item = num_item[np.newaxis,np.newaxis]
            num_array = np.append(num_array, num_item, axis=1)

        for j in range(0, net.number_of_nodes()): 
            #Calculatng denominator of equation
            denominator = (ev.get_timeseries()[1][-1].T[j] - cusp_all)
            denominator = denominator[np.newaxis,np.newaxis]
            sq_array = np.append(sq_array, np.square(denominator), axis=1)  


        sq_sum = np.sum(sq_array, axis = 1)
        num_sum = np.sum(num_array, axis = 1)

        #Calculate correlation
        corr_item = (net.number_of_nodes() * num_sum) / (len(list(neighbour)) * sq_sum)
        corr.append(corr_item)

        # Calculte the average state at each c-value for cell 505 / append the state separetly
        # cusp_505 = ev.get_timeseries()[1][-1].T[505]
        cusp_505 = ev.get_timeseries()[1][-1].T[505]
        #print(f"Shape of cusp_505 is:", np.shape(cusp_505))
        cusp_505 = np.mean(cusp_505)
        #print(f"Average state of cusp505 are,",cusp_505)

        cusp_468 = ev.get_timeseries()[1][-1].T[468]
        cusp_468 = np.mean(cusp_468)

        cusp_487 = ev.get_timeseries()[1][-1].T[505]
        cusp_487 = np.mean(cusp_487)


        # Makes more sense to get the last state of the cusp and plot it
        # cusp_505 = ev.get_timeseries()[1][-1].T[505]
        # print(f"The mean status of cusp0 is:", cusp0_av)
        cusp505_av.append(cusp_505) # to plot the status of cusp0 and cusp1 separetly
        cusp468_av.append(cusp_468)
        cusp487_av.append(cusp_487)

    conv_time = ev.get_timeseries()[0][-1] - ev.get_timeseries()[0][0]
    return conv_time, corr, cusp505_av, c_lis, cusp468_av, cusp487_av, kendall
    # return conv_time, num_tipped, tipped

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


#Network is created using the monthly data, the critical mcwd, coupling switch and the rain factor
net = generate_network(data_crit, data_crit_std, data_eval, no_cpl_dummy, rain_fact, adapt_fact)


###MAIN - PREPARATION###
output = []

init_state = np.zeros(net.number_of_nodes())
init_state.fill(-1) #initial state should be -1 instead of 0 everywhere; this means amazon is covered with veg.

#Without the source node tipped
info = tip_cor(net, init_state)
conv_time = info[0]
correlation = info[1] 
av_505 = info[2]
c_list = info[3]
av_468 = info[4]
av_487 = info[5]
kendallcoef = info[6]
id = f"468, 487, 505"
noise = f"normal"

# Print out tau kendall value for the whole time series
kend_item = kendalltau(c_list, correlation)
print(kend_item)


# nochmal os.chdir neu definieren um outputs/errors abzuspeichern
os.chdir("/p/projects/dominoes/lena/jobs/results/noise/correlation/no_coupling")

'''
if no_cpl_dummy == True:
    np.savetxt("no_coupling/unstable_amaz_{}_{}_adaptsample{}_adaptsigma{}_field_number{}_{}_100_0.01rate_noise{}.txt".format(resolution_type, 
        year_type, str(start_file).zfill(3), int(np.around(100*adapt)), id, float(rain_fact), noise), variance)
    np.savetxt("no_coupling/unstable_amaz_{}_{}_adaptsample{}_adaptsigma{}_total_number{}_{}_100_0.01rate_noise{}.txt".format(resolution_type, 
        year_type, str(start_file).zfill(3), int(np.around(100*adapt)), id, float(rain_fact), noise), [c_list, av_505])
else:
    np.savetxt("unstable_amaz_{}_{}_adaptsample{}_adaptsigma{}_field_number{}_{}_200_0.01rate_noise{}.txt".format(resolution_type, 
        year_type, str(start_file).zfill(3), int(np.around(100*adapt)), id, float(rain_fact), noise), variance)
    np.savetxt("unstable_amaz_{}_{}_adaptsample{}_adaptsigma{}_total_number{}_{}_200_0.01rate_noise{}.txt".format(resolution_type, 
        year_type, str(start_file).zfill(3), int(np.around(100*adapt)), id, float(rain_fact), noise), [c_list, av_505])
'''

fig = plt.figure(figsize = (8,6))
ax1 = fig.add_subplot(111)
ax1.set_ylim(-1.0,1.4)
line1, = ax1.plot(c_list, av_505, 'b', label = "Average state for cusp 505")
line2, = ax1.plot(c_list, av_468, 'y', label = "Average state for cusp 468")
line3, = ax1.plot(c_list, av_487, 'r', label = "Average state for cusp 487")
ax1.set_xlabel('C-values for cusps 468, 487, 505')
ax1.set_ylabel('Average state for forced cusps', color = 'black')
ax1.tick_params(axis='x', labelsize=8)
ax1.tick_params(axis='y', labelsize = 8)

ax2 = ax1.twinx()
line4, = ax2.plot(c_list, correlation, 'g', label = "Spatial correlation")
ax2.set_ylabel('Spatial correlation', color = 'g' )
ax2.tick_params(axis='y', labelsize = 8)
axes = plt.gca()
axes.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
ax1.xaxis.label.set_size(10)
ax1.yaxis.label.set_size(10)
ax2.yaxis.label.set_size(10)
plt.legend((line1, line2, line3, line4), ('Average state for cusp 505', 'Average state for cusp 468', 'Average state for cusp 487', 'Spatial correlation'), prop={'size': 8}, loc='upper left')

#plt.title("Varying c-values for cusps 468, 487 and 505, selected network upper right (0.01*60 rate)", fontsize=10)
plt.title("Varying c-values for cusps 468, 487, 505 with gaussian noise, whole network", fontsize=10)
plt.tight_layout()
fig.savefig("cor_unstable_amaz_{}_{}_adaptsample{}_adaptsigma{}_field_number{}_{}_200__noise{}.png".format(resolution_type, 
        year_type, str(start_file).zfill(3), int(np.around(100*adapt)), id, float(rain_fact), noise), dpi=200)