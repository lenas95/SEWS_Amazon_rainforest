import time
import numpy as np
import networkx as nx
import glob
import re
import os
import itertools
import functools

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

"This code can be used to get the state values as .txt files for the different experiments."
"Furthermore, plots of the regions of the Amazon rainforest to depict the state value distribution of the state values"

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
#neighbour = np.loadtxt("./jobs/results/noise/neighbourslist.txt", dtype=int)

#Load c_values to compute correlation and variance between cells from amzon.py related to the hydro_year
# c_values = np.loadtxt("./jobs/results/noise/final/c_end_values.txt", delimiter=" ",usecols=np.arange(1, n_cols))
#c_end = np.loadtxt("./jobs/results/noise/final/c_end_values.txt", usecols = (1), dtype= np.float64)
#Get all negavtive c-values converted to 0
#c_end[c_end < 0] = 0

#Change directory
c_begin = np.loadtxt("./jobs/results/noise/final/c_begin_values.txt", usecols = 1, dtype = np.float64)

#c_values = np.loadtxt("./jobs/results/noise/final/c_end_values.txt", dtype=dtype)

# Choose betwen NWS(0), NSA(1), SAM(2), NES(3)
region = 3

if region == 0:
    cells = np.loadtxt("./text_files/NWS_cells.txt", dtype=int)
    c_end = np.loadtxt("./text_files/c_end_values_{}_NWS.txt".format(year, year), usecols = (1), dtype= np.float64)
    neighbour = np.loadtxt("./text_files/neighbourslist_NWS.txt", dtype=int)
elif region == 1:
    cells = np.loadtxt("./text_files/NSA_cells.txt", dtype=int)
    c_end = np.loadtxt("./text_files/c_end_values_{}_NSA.txt".format(year,year), usecols = (1), dtype= np.float64)
    neighbour = np.loadtxt("./text_files/neighbourslist_NSA.txt", dtype=int)
elif region == 2:
    cells = np.loadtxt("./text_files/SAM_cells.txt", dtype=int)
    c_end = np.loadtxt("./text_files/c_end_values_{}_SAM.txt".format(year, year), usecols = (1), dtype= np.float64)
    neighbour = np.loadtxt("./text_files/neighbourslist_SAM.txt", dtype=int)
elif region == 3:
    cells = np.loadtxt("./text_files/NES_cells.txt", dtype=int)
    c_end = np.loadtxt("./text_files/c_end_values_{}_NES.txt".format(year, year), usecols = (1), dtype= np.float64)
    neighbour = np.loadtxt("./text_files/neighbourslist_NES.txt", dtype=int)
else:
    print(f"Whole network is selected")
    c_end = np.loadtxt("./text_files/c_end_values_{}.txt".format(year), usecols = (1), dtype= np.float64)
    neighbour = np.loadtxt("./text_files/neighbourslist.txt", dtype=int)
    cells = len(list(0, 567))


c_end[ c_end < 0] = 0
#Try out not including the c>2 values for 2005
#c_end[c_end >= 2.0] = 0
t_step = 0.1
realtime_break = 100 #change to 200 if 2000 timesteps is used in the experiments
timesteps = (realtime_break/t_step)
dc = (c_end/timesteps)

#Experimental setup to get state values

def tip( net , initial_state ):
    
    #ev = evolve( net , initial_state )
    tolerance = 0.01
    #cells = range(0, net.number_of_nodes())
    print(f"Cells are", cells)

    '''
    if not ev.is_equilibrium(tolerance):
        #This network is not in equilibrium since it[0] (= moisture transport) is not effectless at the current state of the network,
      print("Warning: Initial state is not a fixed point of the system")
    elif not ev.is_stable():
        print("Warning: Initial state is not a stable point of the system")
    '''

    # Introduce same noise for all the elements of the network
    # noise = 0.01 # used the same as in my_r_crit
    # sigma = np.diag([1] * net.number_of_nodes()) * noise

    # Create zero arrays to store data

    # cusp_array = np.empty((realtime_break+1,net.number_of_nodes()))
    # cusp_array[:, :] = np.nan
    c_lis = []
    i_lis = []

    #Create empty array to store values of last nodes in a text file
    # ev_array = np.empty((len(list(cells))+1,1))

    os.chdir("/p/projects/dominoes/lena")

    #Load no_coupling and coupling variables for given year and region
    ilist_nocpl = np.loadtxt("./jobs/results/noise/final/{}/no_coupling/i_list_{}_region{}.txt".format(year, year, region), dtype = int)
    all_states_nocpl = np.loadtxt("./jobs/results/noise/final/{}/no_coupling/states_cusps_{}_region{}.txt".format(year, year, region), dtype = np.float64)
    ilist = np.loadtxt("./jobs/results/noise/final/{}/i_list_{}_region{}.txt".format(year, year, region), dtype = int)
    all_states = np.loadtxt("./jobs/results/noise/final/{}/states_cusps_{}_region{}.txt".format(year, year, region), dtype = np.float64)
    
    #for i in range(0, realtime_break*10):
    #for i in range(0, 301):
    
    
    # noise = np.linspace(0.105, 0.200, 20)
    # noise = [0.075, 0.085, 0.095]

    # noise = np.arange(0.075, 0.105, 0.005)
    # noise = [0.095]
    # for noise_val in noise:
    # noise_val = [0.08, 0.09]
    initial_state = np.zeros(net.number_of_nodes())    
    initial_state.fill(-1)
    net = generate_network(data_crit, data_crit_std, data_eval, no_cpl_dummy, rain_fact, adapt_fact)
    ev = evolve( net , initial_state )

    
    #Create empty array to store values of last nodes in a text file
    ev_array = np.empty((len(list(cells))+1,1))
    print(f"Shape of ev_array is", np.shape(ev_array))
    
    for i in itertools.count():
        c_lis = []
        if net.get_tip_states(ev.get_timeseries()[1][-1])[:].any() == False:  
        
            #sigmas = np.array([noise_val] * np.ones(len(list(cells))))
            #alphas = 2 * np.ones(len(list(cells)))
            noise_val = 0.01
            sigmas = np.array([noise_val] * np.ones(len(list(cells))))
            alphas = 2 * np.ones(len(list(cells)))

            for x in range(0, len(list(cells))):
                #if net.nodes[x]['data'].get_par()['c'] <= c_end[x]:
                c = net.nodes[cells[x]]['data'].get_par()['c']
                # print(f"C-Values are,", c)
                net.set_param(cells[x], 'c', c+dc[x])
                c_lis.append(c)

            i_lis.append(i)

            c_lis = [0 if i < 0 else i for i in c_lis]       

            #if functools.reduce(lambda x, y : x and y, map(lambda p, q: p <= q, c_lis, c_end), True): 

            # print(f"Value for c of cell 505:", c)
            ev = evolve (net, ev.get_timeseries()[1][-1])   # making new state as dynamical system for every increased c value
            
            #After evolve the sigmas come in

            #sigmas = np.random.uniform(low=0, high=0.01, size=net.number_of_nodes())
            #sigmas[sigmas < 0] = 0.0
            
            ev.integrate(t_step, realtime_break, sigmas = sigmas, noise = "normal", alphas = alphas, seed = None)
                    
            #Initialize arrays to fill in according to Donangelo et al., spatial variance formula
            sq_array = np.empty((1, 0))
            av_array = np.empty((1, 0))

            #Get c-value at which the first cell tipps
            print(f"Value of c for first tipped in spat var is:", c if net.get_tip_states(ev.get_timeseries()[1][-1])[:].any() != False else "None")
            print(f"First tipped cell are:", net.get_tip_states(ev.get_timeseries()[1][-1])[:])

            arr = np.array(1)
            #Append all nodes last state and save it to .txt file
            for j in range(0, len(list(cells))): 
                cusp_all = ev.get_timeseries()[1][-1].T[cells[j]]
                #print(f"Value of cell j is", ev.get_timeseries()[1][-1].T[cells[j]])
                cusp_all = np.mean(cusp_all)
                #cusp_all = cusp_all[np.newaxis, np.newaxis]
                #print(f"Sahpe of cusp_all is", np.shape(cusp_all))
                arr = np.append(arr, cusp_all)

                #print(f"Shaoe of array looks like", np.shape(arr))
                #print(f"Shape of ev_array is:", np.shape(ev_array))
            
            #ev_array = np.concatenate([ev_array, arr], axis=1)

            arr = arr[: ,np.newaxis]
            #print(f"Shape of new axis ", np.shape(arr))
            ev_array = np.append(ev_array, arr, axis=1)
            #print(f"Shape of ev_array is", np.shape(ev_array))
            #print(f"Ev-Array looks like:", ev_array)    
        
        else:
            break

        
    ev_array = ev_array[1:, 1:]
    print(f"i_lis has shape", np.shape(i_lis))
    print(f"I_list is", i_lis)

    print(f"Ev_Array final has shape:", np.shape(ev_array))
    print(f"First column of ev_array is:", ev_array[:, 0])
    
    
    #Save values for individual years
    # path = '/p/projects/dominoes/lena/jobs/results/noise/final/{}/noise_states'.format(year, region)
    path = '/p/projects/dominoes/lena/jobs/results/noise/final/{}'.format(year)
    fmt = '%1.9f'
    if no_cpl_dummy == True:
        #np.savetxt(os.path.join(path, "no_coupling/region{}/states_cusps_{}_region{}_noise{}.txt".format(region, year, region, noise_val)), ev_array, fmt = fmt)
        np.savetxt(os.path.join(path, "no_coupling/states_cusps_{}_region{}_2000.txt".format(year, region)), ev_array, fmt = fmt)
        np.savetxt(os.path.join(path, "no_coupling/i_list_{}_region{}_2000.txt".format(year, region)), i_lis, fmt = fmt)
    else:
        #np.savetxt(os.path.join(path, "region{}/states_cusps_{}_region{}_noise{}.txt".format(region, year, region, noise_val)), ev_array, fmt = fmt)
        np.savetxt(os.path.join(path, "states_cusps_{}_region{}_2000.txt".format(year, region)), ev_array, fmt = fmt)
        np.savetxt(os.path.join(path, "i_list_{}_region{}_2000.txt".format(year, region)), i_lis, fmt = fmt)

    conv_time = ev.get_timeseries()[0][-1] - ev.get_timeseries()[0][0]
    return conv_time, ev_array, i_lis


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

#Tipping function for correlation
info = tip(net, init_state)
conv_time = info[0]
cusp_all = info[1]
i_list = info[2]


id = f"dc_500"
noise = f"normal"

#Code to load pictures and create plots to depict state value distribution

'''
#Print a picture every 10 iterations
if i == (ilist[-1]/100): #i % 10 == 0 or 
    print(i)
    
    os.chdir("/p/projects/dominoes/lena/jobs/results/noise/final/{}/correlation_status".format(year))
    print("Plotting sequence")
    #latlon values
    lat = net_data.variables["lat"][:]
    lon = net_data.variables["lon"][:]   
    tuples = [(lat[idx],lon[idx]) for idx in range(lat.size)]   
    
    #tuples_2 = [(idx, idy) for idx, idy in range(lat.size)]
    #print(tuples_2)

    #tuples_NWS = [(lat, lon) for lat in range(-17, +11) for lon in range(-79, -71)]
    #print(np.sort(tuples_NWS))
    #tuples_NSA = [(lat, lon) for lat in range(-7, +11) for lon in range(-71, -50)]
    #tuples_SAM = [(lat, lon) for lat in range(-17, -7) for lon in range(-71, -50)]
    #tuples_NES = [(lat, lon) for lat in range(-17, 2) for lon in range(-50, -44)]

    lat = np.unique(lat)
    #print(f"Latitude values are", lat)
    lon = np.unique(lon)
    lat = np.append(lat, lat[-1]+lat[-1]-lat[-2]) 
    lon = np.append(lon, lon[-1]+lon[-1]-lon[-2])

    vals = np.empty((lat.size,lon.size)) #Vals is True or False depending on if the cells tipped
    vals[:,:] = np.nan

    #For different regions latitutde and longitude latitude and longitutde values
    if region == 0:
        tuples_region = [(lat, lon) for lat in range(-17, +11) for lon in range(-79, -71)]                 
        #print(f"Tuple values for region 0 are:", tuples_region)
    elif region == 1:
        tuples_region = [(lat, lon) for lat in range(-7, +11) for lon in range(-71, -50)]
        #print(f"Tuple values for region 0 are:", tuples_region)
    elif region == 2:
        tuples_region = [(lat, lon) for lat in range(-17, -7) for lon in range(-71, -50)]
    elif region == 3:
        tuples_region = [(lat, lon) for lat in range(-17, 2) for lon in range(-50, -44)]
    else:
        lat = lat
        lon = lon

    tuple_list = []
    for idx,x in enumerate(lat):
        for idy,y in enumerate(lon):
            if (x,y) in tuples and (x,y) in tuples_region: #and (x,y) in tuples_region:
                tuple_list.append(tuples.index((x,y)))
                #print(f"(x,y) are", ((x,y)))
                print(f"Tuples index is.,", tuples.index((x,y)))
                cell_index = list(cells).index(tuples.index((x,y)))
                print(f"Cell index is:,", cell_index)

                #Get all cells values if tipped or not in varible p
                #p = net.get_tip_states(ev.get_timeseries()[1][-1])[:][tuples.index((x,y))]
                
                #Get also intermediate values for cells
                if no_cpl_dummy == True:
                    p = all_states_nocpl[cell_index, i]
                    #print(f"p-value of tuple is", p)
                    vals[idx,idy] = p
                else:
                    p = all_states[cell_index, i]
                    vals[idx,idy] = p
            else:
                pass

    print(f"Sorted tuple list is:", np.sort(tuple_list))

    plt.rc('text', usetex=False)
    plt.rc('font', family='serif', size=25)

    plt.figure(figsize=(15,10))

    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([275, 320, -22, 15], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE, linewidth=1)
    ax.coastlines('50m')
    # cmap = plt.get_cmap('turbo')
    # cmap = plt.get_cmap('summer')
    # cmap = matplotlib.cm.ocean(np.linspace(0,1,20))
    cmap = matplotlib.cm.summer(np.linspace(0,1,20))
    cmap = matplotlib.colors.ListedColormap(cmap[0:20])

    plt.pcolor(lon-(lon[-1]-lon[-2]) / 2, lat-(lat[-1]-lat[-2]) / 2, vals, cmap=cmap)
    #nx.draw_networkx(net,pos, edge_color='black', node_size=0, with_labels=False)
    cbar = plt.colorbar(label='Unstable Amazon states with coupling')

    if no_cpl_dummy == True:
        plt.savefig("no_coupling/region{}/unstable_amaz_{}_{}_adaptsample{}_adaptsigma{}_number{}_{}_{}_pic{}.png".format(region, resolution_type, year_type, 
            str(start_file).zfill(3), int(np.around(100*adapt)), id, float(rain_fact), realtime_break, i), bbox_inches='tight')
    else:
        plt.savefig("region{}/unstable_amaz_{}_{}_adaptsample{}_adaptsigma{}_number{}_{}_{}_pic{}.png".format(region, resolution_type, year_type, 
            str(start_file).zfill(3), int(np.around(100*adapt)), id, float(rain_fact), realtime_break, i), bbox_inches='tight')
        

    #plt.show()
    plt.clf()
    plt.close() 
'''
