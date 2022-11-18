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
#from evolve import evolve, NoEquilibrium
from evolve_sde import evolve, NoEquilibrium
from tipping_element import cusp
from coupling import linear_coupling
from functions_amazon import global_functions

"This code can be used to calculate the spatial variance (and plot it against average state of cell) for the whole network when increasing c_value for one cell"

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
    
    ev = evolve(net , initial_state )
    tolerance = 0.01
    

    if not ev.is_equilibrium(tolerance):
        #This network is not in equilibrium since it[0] (= moisture transport) is not effectless at the current state of the network,
      print("Warning: Initial state is not a fixed point of the system")
    elif not ev.is_stable():
        print("Warning: Initial state is not a stable point of the system")

    t_step = 1
    realtime_break = 200 #originally 30000

    # dc = 0.005
    # dc = np.arange(0, 0.7, 0.01)
    dc = np.array([0.01] * 80)  # with dc = np.array([0.01] * 20)  and realtime_break = 200 works for 0.85 tipping cascade and for 1.0 rain_fact use [0.05*20]

    # Introduce same noise for all the elements of the network
    # noise = 0.01 # used the same as in my_r_crit
    # sigma = np.diag([1] * net.number_of_nodes()) * noise

    # sigma = None # To implement no noise

    # cusp_array = np.empty((realtime_break+1,net.number_of_nodes()))
    # cusp_array[:, :] = np.nan

    c_lis = []
    var = []
    cusp505_av = []
    
    for i in range(0, len(dc)):
        
        #Initialize arrays to fill in according to Donangelo et al., spatial variance formula
        # sq_array = np.empty((realtime_break+1, 0))
        # av_array = np.empty((realtime_break+1, 0))
        sq_array = np.empty((1,0))
        av_array = np.empty((1,0))

        c = net.nodes[505]['data'].get_par()['c'] 
        net.set_param(505, 'c', c+dc[i])
        c_lis.append(c) # Append c to list so can be plotted on x-axis against spatial variance

        print(f"Value for c of cell 505:", c)
        
        #ts_f = ev.get_timeseries()[1][198:,:]
        #ts_f[ts_f <= -1.0] = -1.0
        
        ev = evolve(net, ev.get_timeseries()[1][-1]) 
        # ev = evolve(net, ts)
        ts = ev.get_timeseries()[1][-1]
        # ts[ts <= -1.0] = -1.0   #Limit the spatial values to -1.0 to prevent an overshoot of moisture recycling to other cells due to noise
        #print(f"ts is :", ts)
        # [ev.get_timeseries()[1][-1] = -1.0 for x in ev.get_timeseries()[1][-1] if x < -1.0]
        # if ev.get_timeseries()[1][-1] < -1.0  # making new state as dynamical system for every increased c value
        # print(f"ev.timeseries before integrating is,", ev.get_timeseries()[1][-1])
        # ev.get_timeseries()[1][-1][ev.get_timeseries()[1][-1] <= -1.0] = -1.0
        # ts[ts <= 1.0] = -1.0
        #for x in range(0, len(ev.get_timeseries()[1][-1])):
        #    if ev.get_timeseries()[1][-1][x] < -1.0:
        #        ev.get_timeseries()[1][-1][x] = -1.0
        #    else:
        #        pass
        # sigmas = 0.01 * np.ones(net.number_of_nodes())  #this gives straight lines
        sigmas = np.random.random(net.number_of_nodes()) - 0.9
        sigmas[sigmas < 0] = 0.0
        sigmas[505] = 0.01
        alphas = 2 * np.ones(net.number_of_nodes())
        ev.integrate(t_step, realtime_break, sigmas = sigmas, noise = "normal", alphas = alphas, seed = 1)
        # ev.integrate(t_step, realtime_break, ts, sigma=sigma, noise="levy")
        # ev.integrate(t_step, realtime_break, ev.get_timeseries()[1][-1], sigma=sigma)
        #ev.get_timeseries()[1][-1][ev.get_timeseries()[1][-1] <= -1.0] = -1.0
        # ts_f = ev.get_timeseries()[1][198:, :]
        # ts_f[ts_f <= -1.0] = -1.0
        #print(f"ev.timeseries after integrating is,", ev.get_timeseries()[1][-1] )
        # print(f"ts after integrating,",  ts_f)
        #ts = ev.get_timeseries()[1][-1]
        #[ts[item] == -1.0 for item in ts if item < 1.0]

        for j in range(0, net.number_of_nodes()):          # Calculate variance for all nodes
            cusp_el = ev.get_timeseries()[1][-1].T[j]
            # cusp_el = ev.get_timeseries()[1][198:, :].T[j]
            # cusp_el = ts_f.T[j] 

            #Add new axis to append columns so that converts to (ev.get_timeseries(), 1)
            cusp_el = np.mean(cusp_el)
            # cusp_el = cusp_el[...,np.newaxis] 
            cusp_el = cusp_el[np.newaxis,np.newaxis] 

            # Append columns of squared values of cusp_el 
            sq_array = np.append(sq_array, np.square(cusp_el), axis=1)  
            av_array = np.append(av_array, cusp_el, axis = 1)

        #sq_sum = np.sum(np.sum(sq_array, axis = 1), axis=0)
        #av_sum = np.sum(np.sum(av_array, axis = 1), axis=0)

        sq_sum = np.sum(sq_array, axis=1)
        av_sum = np.sum(av_array, axis=1)

        item_var = (sq_sum - (np.square(av_sum))) / (net.number_of_nodes()**2)
        var.append(item_var)

        # Calculte the average state at each c-value for cell 505 / append the state separetly
        cusp_505 = ev.get_timeseries()[1][-1].T[505]
        # cusp_505  =  ev.get_timeseries()[1][198:, :].T[505]
        # cusp_505 = ts_f.T[505]
        print(f"Last entries of cusp505 are:", cusp_505)
        cusp_505 = np.mean(cusp_505)
        print(f"mean of cusp505 is", cusp_505)

        # Makes more sense to get the last state of the cusp and plot it
        # cusp_505 = ev.get_timeseries()[1][-1].T[505]
        #print(f"The mean status of cusp0 is:", cusp0_av)
        cusp505_av.append(cusp_505) # to plot the status of cusp0 and cusp1 separetly

    conv_time = ev.get_timeseries()[0][-1] - ev.get_timeseries()[0][0]
    return conv_time, var, cusp505_av, c_lis
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
id = 505
noise = f"normal_np.rand_positivessigmas_505=0.01"

# nochmal os.chdir neu definieren um outputs/errors abzuspeichern
os.chdir("/p/projects/dominoes/lena/jobs/results/noise/spat_variance/levy_noise")

'''
if no_cpl_dummy == True:
    np.savetxt("no_coupling/unstable_amaz_{}_{}_adaptsample{}_adaptsigma{}_field_number{}_{}_200_noise{}_ts.txt".format(resolution_type, 
        year_type, str(start_file).zfill(3), int(np.around(100*adapt)), id, float(rain_fact), noise), variance)
    np.savetxt("no_coupling/unstable_amaz_{}_{}_adaptsample{}_adaptsigma{}_total_number{}_{}_200_noise{}_ts.txt".format(resolution_type, 
        year_type, str(start_file).zfill(3), int(np.around(100*adapt)), id, float(rain_fact), noise), [c_list, av_505])
else:
    np.savetxt("unstable_amaz_{}_{}_adaptsample{}_adaptsigma{}_field_number{}_{}_200_noise{}_ts.txt".format(resolution_type, 
        year_type, str(start_file).zfill(3), int(np.around(100*adapt)), id, float(rain_fact), noise), variance)
    np.savetxt("unstable_amaz_{}_{}_adaptsample{}_adaptsigma{}_total_number{}_{}_200_noise{}_ts.txt".format(resolution_type, 
        year_type, str(start_file).zfill(3), int(np.around(100*adapt)), id, float(rain_fact), noise), [c_list, av_505])
'''

fig = plt.figure(figsize = (8,6))
ax1 = fig.add_subplot(111)
ax1.set_ylim(-1.0,1.4)
line1, = ax1.plot(c_list, av_505, 'b', label = "Average state for cusp 505")
ax1.set_xlabel('C-values for cusp 505')
ax1.set_ylabel('Average state for cusp 505', color = 'b')
ax1.tick_params(axis='x', labelsize=8)
ax1.tick_params(axis='y', labelsize = 8)

ax2 = ax1.twinx()
line2, = ax2.plot(c_list, variance, 'g', label = "Spatial variance")
ax2.set_ylabel('Spatial Variance', color = 'g' )
ax2.tick_params(axis='y', labelsize = 8)
axes = plt.gca()
axes.ticklabel_format(style='sci', axis='both', scilimits=(0,0))
ax1.xaxis.label.set_size(10)
ax1.yaxis.label.set_size(10)
ax2.yaxis.label.set_size(10)
plt.legend((line1, line2), ('Average state for cusp 505', 'Spatial variance'), prop={'size': 5}, loc='upper left')

plt.title("Varying c-values for cusp505 with gaussian noise, whole network", fontsize=10)
plt.tight_layout()
fig.savefig("spat_var_unstable_amaz_{}_{}_adaptsample{}_adaptsigma{}_field_number{}_{}_200_noise{}.png".format(resolution_type, 
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
