#use python3 r_crit_unstable_amazon.py 0 2004 0 1.0 1 
#Chage 2004 to the year you want to examine



# %%


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

"This code can be used to calculate the spatial variance (and plot it against average state of cell) for specially selected cells when increasing c_value for one cell"

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

#Tipping function
def tip_var( net , initial_state ):
    
    ev = evolve( net , initial_state )
    tolerance = 0.01
    

    if not ev.is_equilibrium(tolerance):
        #This network is not in equilibrium since it[0] (= moisture transport) is not effectless at the current state of the network,
      print("Warning: Initial state is not a fixed point of the system")
    elif not ev.is_stable():
        print("Warning: Initial state is not a stable point of the system")

    t_step = 1
    realtime_break = 200 #originally 30000 and works with 200 (see r_crt_unstable_amazon.py)

    # dc = 0.005
    # dc = np.arange(0, 0.7, 0.01)
    dc = np.array([0.01] * 60)  # with dc = np.array([0.01] * 20)  and realtime_break = 200 works for 0.85 tipping cascade and for 1.0: rain_fact use [0.05] *20
    #Doesn't work for 0.01*80 and 100 ;

    # Introduce same noise for all the elements of the network
    noise = 0.01 # used the same as in my_r_crit
    sigma = np.diag([1] * net.number_of_nodes()) * noise


    # Create zero arrays to store data

    # cusp_array = np.empty((realtime_break+1,net.number_of_nodes()))
    # cusp_array[:, :] = np.nan

    c_lis = []
    cells = list(range(446, 454)) + list(range(466, 473)) + list(range(485,492)) + list(range(504, 510)) + list(range(521,525)) + list(range(534, 537)) +list(range(544, 547))
    # cells = list(range(466, 473)) + list(range(485,492)) + list(range(504, 510)) + list(range(521,525)) + list(range(534, 537))
    var = []
    cusp505_av = []
    cusp468_av = []
    cusp487_av = []
    c_cells = [468, 487, 505]
    
    for i in range(0, len(dc)):
        
        #Initialize arrays to fill in according to Donangelo et al., spatial variance formula
        sq_array = np.empty((1, 0))
        av_array = np.empty((1, 0))

        #c = net.nodes[505]['data'].get_par()['c'] 
        #net.set_param(505, 'c', c+dc[i])
        for x in net.number_of_nodes():
            c = net.nodes[x]['data'].get_par()['c'] 
            net.set_param(x, 'c', c+dc[i])
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

        for j in range(0, net.number_of_nodes()):          # Calculate variance for all nodes
            cusp_el = ev.get_timeseries()[1][-1].T[j]
            #cusp_el = ev.get_timeseries()[1][:5, :].T[cells[j]]
            #Add new axis to append columns so that converts to (ev.get_timeseries(), 1)
            cusp_el = np.mean(cusp_el)
            cusp_el = cusp_el[np.newaxis,np.newaxis] 

            # Append columns of squared values of cusp_el 
            sq_array = np.append(sq_array, np.square(cusp_el), axis=1)  
            print(f"Shape of sq_array", np.shape(sq_array))
            print(f"The sq_array looks like:", sq_array)
            av_array = np.append(av_array, cusp_el, axis = 1)

        sq_sum = np.sum(sq_array, axis=1)
        av_sum = np.sum(av_array, axis=1)

        item_var = (sq_sum - (np.square(av_sum))) / (net.number_of_nodes()**2)
        var.append(item_var)
        #print(f"list of variance items is:", var)

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
    return conv_time, var, cusp505_av, c_lis, cusp468_av, cusp487_av 
    # return conv_time, num_tipped, tipped

###MAIN - PREPARATION###
#need changing variables from file names
dataset = data_crit[0]
net_data = Dataset(dataset)

#latlon values and c values
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
info = tip_var(net, init_state)
conv_time = info[0]
variance = info[1] 
av_505 = info[2]
c_list = info[3]
av_468 = info[4]
av_487 = info[5]
id = f"all"
noise = f"normal"

# nochmal os.chdir neu definieren um outputs/errors abzuspeichern
os.chdir("/p/projects/dominoes/lena/jobs/results/noise/spat_variance/individual_cells/all_cells")

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
ax1.set_xlabel('C-values for cusp 505')
ax1.set_ylabel('Average state for forced cusps', color = 'black')
ax1.tick_params(axis='x', labelsize=8)
ax1.tick_params(axis='y', labelsize = 8)

ax2 = ax1.twinx()
line4, = ax2.plot(c_list, variance, 'g', label = "Spatial variance")
ax2.set_ylabel('Spatial Variance', color = 'g' )
ax2.tick_params(axis='y', labelsize = 8)
axes = plt.gca()
axes.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
ax1.xaxis.label.set_size(10)
ax1.yaxis.label.set_size(10)
ax2.yaxis.label.set_size(10)
plt.legend((line1, line2, line3, line4), ('Average state for cusp 505', 'Average state for cusp 468', 'Average state for cusp 487', 'Spatial variance'), prop={'size': 8}, loc='upper left')

#plt.title("Varying c-values for cusps 468, 487 and 505, selected network upper right (0.01*60 rate)", fontsize=10)
plt.title("Varying c-values for all cells with gaussian noise whole network, all cells forced", fontsize=10)
plt.tight_layout()
fig.savefig("spat_var_unstable_amaz_{}_{}_adaptsample{}_adaptsigma{}_field_{}_{}_200__noise{}_allstates.png".format(resolution_type, 
        year_type, str(start_file).zfill(3), int(np.around(100*adapt)), id, float(rain_fact), noise), dpi=200)


'''
#plotting procedure

print("Plotting sequence")
tuples = [(lat[idx],lon[idx]) for idx in range(lat.size)]

lat = np.unique(lat)
lon = np.unique(lon)
lat = np.append(lat, lat[-1]+lat[-1]-lat[-2]) #why do we need to append lat[-1]+lat[-1]-lat[-2]???
lon = np.append(lon, lon[-1]+lon[-1]-lon[-2])
vals = np.empty((lat.size,lon.size))
vals[:,:] = np.nan

for idx,x in enumerate(lat):
    for idy,y in enumerate(lon):
        if (x,y) in tuples:
            p = unstable_amaz[tuples.index((x,y))]
            vals[idx,idy] = p


plt.rc('text', usetex=False)
plt.rc('font', family='serif', size=25)

plt.figure(figsize=(15,10))

ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_extent([275, 320, -22, 15], crs=ccrs.PlateCarree())
ax.add_feature(cfeature.COASTLINE, linewidth=1)
ax.coastlines('50m')
#cmap = plt.get_cmap('turbo')
cmap = plt.get_cmap('summer')

plt.pcolor(lon-(lon[-1]-lon[-2]) / 2, lat-(lat[-1]-lat[-2]) / 2, vals, cmap=cmap)
#nx.draw_networkx(net,pos, edge_color='black', node_size=0, with_labels=False)
cbar = plt.colorbar(label='Unstable Amazon')
if no_cpl_dummy == True:
    plt.savefig("r_crit/no_coupling/unstable_amaz_{}_{}_adaptsample{}_adaptsigma{}.png".format(resolution_type, year_type, 
        str(start_file).zfill(3), int(np.around(100*adapt))), bbox_inches='tight')
    plt.savefig("r_crit/no_coupling/unstable_amaz_{}_{}_adaptsample{}_adaptsigma{}.pdf".format(resolution_type, year_type, 
        str(start_file).zfill(3), int(np.around(100*adapt))), bbox_inches='tight')
else:
    plt.savefig("unstable_amaz_{}_{}_adaptsample{}_adaptsigma{}_number{}_{}_200_noise{}.png".format(resolution_type, year_type, 
        str(start_file).zfill(3), int(np.around(100*adapt)), id, float(rain_fact), noise), bbox_inches='tight')
    plt.savefig("unstable_amaz_{}_{}_adaptsample{}_adaptsigma{}_number{}_{}_200_noise{}.pdf".format(resolution_type, year_type, 
        str(start_file).zfill(3), int(np.around(100*adapt)), id, float(rain_fact), noise), bbox_inches='tight')

#plt.show()
plt.clf()
plt.close()
'''

print("Finish")
# %%
