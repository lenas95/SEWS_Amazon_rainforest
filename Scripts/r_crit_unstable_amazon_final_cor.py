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
import itertools

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



#Load c_values to compute correlation and variance between cells from amzon.py related to the hydro_year

#c_end = np.loadtxt("./jobs/results/noise/final/c_end_values.txt", usecols = (1), dtype= np.float64)
#Get all negavtive c-values converted to 0
#c_end[c_end < 0] = 0
#dc = (c_end/500)

c_begin = np.loadtxt("./jobs/results/noise/final/c_begin_values.txt", usecols = 1, dtype = np.float64)

#Load neighbourlist to compute correlation between cells from r_crit_unstable_amazon.py
neighbour = np.loadtxt("./jobs/results/noise/neighbourslist.txt", dtype=int)

# Choose between NWS(0), NSA(1), SAM(2), NES(3)
region = 0

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
    neighbour = np.loadtxt("./jobs/results/noise/neighbourslist.txt", dtype=int)

c_end[ c_end < 0] = 0
t_step = 0.1
realtime_break = 100 #originally 30000 and works with 200 (see r_crt_unstable_amazon.py)
timesteps = (realtime_break/t_step)
dc = (c_end/timesteps)

#Tipping function for plotting correlation 
def tip_cor( net , initial_state ):
    
    ev = evolve( net , initial_state )
    tolerance = 0.01
    

    if not ev.is_equilibrium(tolerance):
        #This network is not in equilibrium since it[0] (= moisture transport) is not effectless at the current state of the network,
      print("Warning: Initial state is not a fixed point of the system")
    elif not ev.is_stable():
        print("Warning: Initial state is not a stable point of the system")

    t_step = 0.1
    realtime_break = 100 #originally 30000 and works with 200 (see r_crt_unstable_amazon.py)

    # dc = np.array([0.01] * 60)  # with dc = np.array([0.01] * 20)  and realtime_break = 200 works for 0.85 tipping cascade and for 1.0: rain_fact use [0.05] *20

    # Create zero arrays to store data

    i_lis = []
    # cells = list(range(446, 454)) + list(range(466, 473)) + list(range(485,492)) + list(range(504, 510)) + list(range(521,525)) + list(range(534, 537)) +list(range(544, 547))
    # cells = list(range(466, 473)) + list(range(485,492)) + list(range(504, 510)) + list(range(521,525)) + list(range(534, 537))
    corr = []
    
    # Tipping cells for 0,1,2,3 (2005)
    cusp47_av = [] #For region 0
    cusp172_av = [] #For region 1
    cusp538_av = [] #For region 3

    cusp400_av = []
    cusp401_av = []
    cusp420_av = []
    cusp421_av = []
    cusp440_av = []
    cusp441_av = []

    # Tipping cells for 0,1,3 (2007) Region 2 is 440
    cusp30_av = []
    cusp218_av = []
    cusp540_av = []

    # Tipping cells for 1 Region (2010) Region 2 is 400, Region 3 538 and Region 0 30
    cusp216_av = []
    
    
    #for i in range(0, (realtime_break*10)):
    for i in itertools.count():
        if net.get_tip_states(ev.get_timeseries()[1][-1])[:].any() == False: 
            #Initialize arrays to fill in according to Dakos et al. 2010, correlation formula
            sq_array = np.empty((1, 0))
            num_array = np.empty((1, 0))
            # cusp_all_arr = np.empty((1,0))
            cusp_all_arr = []

            #c = net.nodes[505]['data'].get_par()['c']  # When only one cell is forced uncomment
            #net.set_param(505, 'c', c+dc[i])

            #for x in range(0, net.number_of_nodes()):
            for x in range(0, len(list(cells))):
                #if net.nodes[x]['data'].get_par()['c'] <= c_end[x]:
                c = net.nodes[cells[x]]['data'].get_par()['c']
                # print(f"C-Values are,", c)
                net.set_param(cells[x], 'c', c+dc[x])
            i_lis.append(i)

            ev = evolve (net, ev.get_timeseries()[1][-1])   # making new state as dynamical system for every increased c value
            
            #sigmas = np.array([0.01] * np.ones(net.number_of_nodes()))
            #alphas = 2 * np.ones(net.number_of_nodes())

            sigmas = np.array([0.01] * np.ones(len(list(cells))))
            alphas = 2 * np.ones(len(list(cells)))
            ev.integrate(t_step, realtime_break, sigmas = sigmas, noise = "normal", alphas = alphas, seed = None)

            #Get i-value at which the first cell tipps
            print(f"Value of i for first tipped in correlation is:", i if net.get_tip_states(ev.get_timeseries()[1][-1])[:].any() != False else "None")
            # print(f"Value of cell for first tipped in correlation is:", x if net.get_tip_states(ev.get_timeseries()[1][-1])[:].any() != False else "None")
            
            #When plotting whole netowrk over several iterations beyond first tipped append c_corr
            #if net.get_tip_states(ev.get_timeseries()[1][-1])[:].any() != False:
            #    c_corr.append(i)
            #print(c_corr)
            
            #Get the average of last state for all cusps 
            for x in range (0, len(list(cells))):
                cusp_all = ev.get_timeseries()[1][-1].T[cells[x]]
                #cusp_all = cusp_all[np.newaxis,np.newaxis]
                # cusp_all_arr.append(cusp_all_arr, cusp_all, axis =1)
                cusp_all_arr.append(cusp_all)

            print(f"cusp alls array is:", cusp_all_arr)
            #cusp_all = np.mean(cusp_all_arr) 
            #cusp_all = (ev.get_timeseries()[1].T[x] for x in list(cells))
            cusp_all = np.mean(cusp_all_arr)
            print(f"Cusp_all is:", cusp_all)

            for x in range(0, len(list(neighbour))):
                
                #Get all the values for one c for the numerator
                num_item = (ev.get_timeseries()[1][-1].T[neighbour[x,0]] - cusp_all) * (ev.get_timeseries()[1][-1].T[neighbour[x,1]] - cusp_all)
                num_item = num_item[np.newaxis,np.newaxis]
                num_array = np.append(num_array, num_item, axis=1)

            #for j in range(0, net.number_of_nodes()): 
            for j in range(0, len(list(cells))):
                #Calculatng denominator of equation
                denominator = (ev.get_timeseries()[1][-1].T[j] - cusp_all)
                denominator = denominator[np.newaxis,np.newaxis]
                sq_array = np.append(sq_array, np.square(denominator), axis=1)  


            sq_sum = np.sum(sq_array, axis = 1)
            num_sum = np.sum(num_array, axis = 1)

            #Calculate correlation
            #corr_item = (net.number_of_nodes() * num_sum) / (len(list(neighbour)) * sq_sum)
            corr_item = (len(list(cells)) * num_sum) / (len(list(neighbour)) * sq_sum)
            corr.append(corr_item)

            # Calculte the average state at each c-value for tipped cells by regions / append the state separetly
            # Year 2005
            cusp_538 = ev.get_timeseries()[1][-1].T[538]
            cusp_47 = ev.get_timeseries()[1][-1].T[47]
            cusp_172 = ev.get_timeseries()[1][-1].T[172]

            cusp_400 = ev.get_timeseries()[1][-1].T[400]
            cusp_401 = ev.get_timeseries()[1][-1].T[401]
            cusp_420 = ev.get_timeseries()[1][-1].T[420]
            cusp_421 = ev.get_timeseries()[1][-1].T[421]
            cusp_440 = ev.get_timeseries()[1][-1].T[440]
            cusp_441 = ev.get_timeseries()[1][-1].T[441]

            cusp538_av.append(cusp_538) #Region3
            cusp47_av.append(cusp_47)  #Region0
            cusp172_av.append(cusp_172)  #Region1

            cusp400_av.append(cusp_400)  # For 2005 (Region 2) and 2007 (also Region 2)
            cusp401_av.append(cusp_401)
            cusp420_av.append(cusp_420)
            cusp421_av.append(cusp_421)  
            cusp440_av.append(cusp_440)
            cusp441_av.append(cusp_441)

            #Year 2007
            cusp_30 = ev.get_timeseries()[1][-1].T[30]
            cusp_218 = ev.get_timeseries()[1][-1].T[218]
            cusp_540 = ev.get_timeseries()[1][-1].T[540]

            cusp30_av.append(cusp_30) #Region0
            cusp218_av.append(cusp_218)  #Region1
            cusp540_av.append(cusp_540) #Region3

            #Year 2010
            cusp_216 = ev.get_timeseries()[1][-1].T[216]
            cusp216_av.append(cusp_216)

        else:
            break

    conv_time = ev.get_timeseries()[0][-1] - ev.get_timeseries()[0][0]
    return conv_time, corr, i_lis, cusp400_av, cusp401_av, cusp420_av, cusp421_av, cusp440_av, cusp441_av, cusp47_av, cusp172_av, cusp538_av, cusp30_av, cusp218_av, cusp540_av, cusp216_av
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

#Tipping function for correlation
info_cor = tip_cor(net, init_state)
conv_time = info_cor[0]
correlation = info_cor[1] 
i_list = info_cor[2]
av_400 = info_cor[3]
av_401 = info_cor[4]
av_420 = info_cor[5]
av_421 = info_cor[6]
av_440 = info_cor[7]
av_441 = info_cor[8]
# c_lis = info_var[5]
av_47 = info_cor[9]
av_172 = info_cor[10]
av_538 = info_cor[11]
av_30 = info_cor[12]
av_218 = info_cor[13]
av_540 = info_cor[14]
av_216 = info_cor[15]


i_list_new = [item * 100 for item in i_list] #To get time and not iterations
list(i_list_new)

id = "dc{}_region{}".format(timesteps, region)
noise = f"normal"

# Print out tau kendall value for the whole time series
#kend_item = kendalltau(c_list, correlation)
#print(kend_item)


# nochmal os.chdir neu definieren um outputs/errors abzuspeichern
os.chdir("/p/projects/dominoes/lena/jobs/results/noise/final/{}/correlation".format(year))

#Save correlation and forced cusps for no_coupling to plot them together
if no_cpl_dummy == True and int(year) == 2005:
    print(f"Year 2005 is selected")
    if region == 0:
        np.savetxt("no_coupling/correlation_no_coupling{}_region{}.txt".format(year, region), correlation, fmt = '%1.9f')
        np.savetxt("no_coupling/cell_no_coupling{}_region{}.txt".format(year, region), av_47, fmt = '%1.9f')
        np.savetxt("no_coupling/ilist_no_coupling{}_region{}.txt".format(year, region), i_list_new, fmt = '%d')
    elif region == 1:
        np.savetxt("no_coupling/correlation_no_coupling{}_region{}.txt".format(year, region), correlation, fmt = '%1.9f')
        np.savetxt("no_coupling/cell_no_coupling{}_region{}.txt".format(year, region), av_172, fmt = '%1.9f')
        np.savetxt("no_coupling/ilist_no_coupling{}_region{}.txt".format(year, region), i_list_new, fmt = '%d')
    elif region == 2:
        np.savetxt("no_coupling/correlation_no_coupling{}_region{}.txt".format(year, region), correlation, fmt = '%1.9f')
        np.savetxt("no_coupling/cell_no_coupling{}_region{}_400.txt".format(year, region), av_400, fmt = '%1.9f')
        np.savetxt("no_coupling/cell_no_coupling{}_region{}_401.txt".format(year, region), av_401, fmt = '%1.9f')
        np.savetxt("no_coupling/cell_no_coupling{}_region{}_420.txt".format(year, region), av_420, fmt = '%1.9f')
        np.savetxt("no_coupling/cell_no_coupling{}_region{}_421.txt".format(year, region), av_421, fmt = '%1.9f')
        np.savetxt("no_coupling/cell_no_coupling{}_region{}_440.txt".format(year, region), av_440, fmt = '%1.9f')
        np.savetxt("no_coupling/cell_no_coupling{}_region{}_441.txt".format(year, region), av_441, fmt = '%1.9f')
        np.savetxt("no_coupling/ilist_no_coupling{}_region{}.txt".format(year, region), i_list_new, fmt = '%d')
    elif region == 3:
        np.savetxt("no_coupling/correlation_no_coupling{}_region{}.txt".format(year, region), correlation, fmt = '%1.9f')
        np.savetxt("no_coupling/cell_no_coupling{}_region{}.txt".format(year, region), av_538, fmt = '%1.9f')
        np.savetxt("no_coupling/ilist_no_coupling{}_region{}.txt".format(year, region), i_list_new, fmt = '%d')
    else:
        pass


if no_cpl_dummy == True and int(year) == 2007:
    print(f"Year 2007 is selected")
    if region == 0:
        np.savetxt("no_coupling/correlation_no_coupling{}_region{}.txt".format(year, region), correlation, fmt = '%1.9f')
        np.savetxt("no_coupling/cell_no_coupling{}_region{}.txt".format(year, region), av_30, fmt = '%1.9f')
        np.savetxt("no_coupling/ilist_no_coupling{}_region{}.txt".format(year, region), i_list_new, fmt = '%d')
    elif region == 1:
        np.savetxt("no_coupling/correlation_no_coupling{}_region{}.txt".format(year, region), correlation, fmt = '%1.9f')
        np.savetxt("no_coupling/cell_no_coupling{}_region{}.txt".format(year, region), av_218, fmt = '%1.9f')
        np.savetxt("no_coupling/ilist_no_coupling{}_region{}.txt".format(year, region), i_list_new, fmt = '%d')
    elif region == 2:
        np.savetxt("no_coupling/correlation_no_coupling{}_region{}.txt".format(year, region), correlation, fmt = '%1.9f')
        np.savetxt("no_coupling/cell_no_coupling{}_region{}_400.txt".format(year, region), av_400, fmt = '%1.9f')
        np.savetxt("no_coupling/cell_no_coupling{}_region{}_401.txt".format(year, region), av_401, fmt = '%1.9f')
        np.savetxt("no_coupling/ilist_no_coupling{}_region{}.txt".format(year, region), i_list_new, fmt = '%d')
    elif region == 3:
        np.savetxt("no_coupling/correlation_no_coupling{}_region{}.txt".format(year, region), correlation, fmt = '%1.9f')
        np.savetxt("no_coupling/cell_no_coupling{}_region{}.txt".format(year, region), av_540, fmt = '%1.9f')
        np.savetxt("no_coupling/ilist_no_coupling{}_region{}.txt".format(year, region), i_list_new, fmt = '%d')
    else:
        pass

if no_cpl_dummy == True and int(year) == 2010:  
    print(f"Year 2010 is selected")      
    if region == 0:
        np.savetxt("no_coupling/correlation_no_coupling{}_region{}.txt".format(year, region), correlation, fmt = '%1.9f')
        np.savetxt("no_coupling/cell_no_coupling{}_region{}.txt".format(year, region), av_30, fmt = '%1.9f')
        np.savetxt("no_coupling/ilist_no_coupling{}_region{}.txt".format(year, region), i_list_new, fmt = '%d')
    elif region == 1:
        np.savetxt("no_coupling/correlation_no_coupling{}_region{}.txt".format(year, region), correlation, fmt = '%1.9f')
        np.savetxt("no_coupling/cell_no_coupling{}_region{}.txt".format(year, region), av_216, fmt = '%1.9f')
        np.savetxt("no_coupling/ilist_no_coupling{}_region{}.txt".format(year, region), i_list_new, fmt = '%d')
    elif region == 2:
        np.savetxt("no_coupling/correlation_no_coupling{}_region{}.txt".format(year, region), correlation, fmt = '%1.9f')
        np.savetxt("no_coupling/cell_no_coupling{}_region{}_400.txt".format(year, region), av_400, fmt = '%1.9f')
        np.savetxt("no_coupling/cell_no_coupling{}_region{}_420.txt".format(year, region), av_420, fmt = '%1.9f')
        np.savetxt("no_coupling/cell_no_coupling{}_region{}_440.txt".format(year, region), av_440, fmt = '%1.9f')
        np.savetxt("no_coupling/ilist_no_coupling{}_region{}.txt".format(year, region), i_list_new, fmt = '%d')
    elif region == 3:
        np.savetxt("no_coupling/correlation_no_coupling{}_region{}.txt".format(year, region), correlation, fmt = '%1.9f')
        np.savetxt("no_coupling/cell_no_coupling{}_region{}.txt".format(year, region), av_538, fmt = '%1.9f')
        np.savetxt("no_coupling/ilist_no_coupling{}_region{}.txt".format(year, region), i_list_new, fmt = '%d')
    else:
        pass

#Reload no coupling values saved before when performing calculations with coupling
if int(year) == 2005:
    if region == 0:
        no_cpl_correlation = np.loadtxt("no_coupling/correlation_no_coupling{}_region{}.txt".format(year,region), dtype= np.float64)
        no_cpl_cell = np.loadtxt("no_coupling/cell_no_coupling{}_region{}.txt".format(year, region), dtype= np.float64)
        ilist_nocpl = np.loadtxt("no_coupling/ilist_no_coupling{}_region{}.txt".format(year, region), dtype = int)
    elif region == 1:
        no_cpl_correlation = np.loadtxt("no_coupling/correlation_no_coupling{}_region{}.txt".format(year,region), dtype= np.float64)
        no_cpl_cell = np.loadtxt("no_coupling/cell_no_coupling{}_region{}.txt".format(year, region), dtype= np.float64)
        ilist_nocpl = np.loadtxt("no_coupling/ilist_no_coupling{}_region{}.txt".format(year, region), dtype = int)
    elif region == 2:
        no_cpl_correlation = np.loadtxt("no_coupling/correlation_no_coupling{}_region{}.txt".format(year,region), dtype= np.float64)
        no_cpl_cell_400 = np.loadtxt("no_coupling/cell_no_coupling{}_region{}_400.txt".format(year, region), dtype= np.float64)
        no_cpl_cell_401 = np.loadtxt("no_coupling/cell_no_coupling{}_region{}_401.txt".format(year, region), dtype= np.float64)
        no_cpl_cell_420 = np.loadtxt("no_coupling/cell_no_coupling{}_region{}_420.txt".format(year, region), dtype= np.float64)
        no_cpl_cell_421 = np.loadtxt("no_coupling/cell_no_coupling{}_region{}_421.txt".format(year, region), dtype= np.float64)
        no_cpl_cell_440 = np.loadtxt("no_coupling/cell_no_coupling{}_region{}_440.txt".format(year, region), dtype= np.float64)
        no_cpl_cell_441 = np.loadtxt("no_coupling/cell_no_coupling{}_region{}_441.txt".format(year, region), dtype= np.float64)
        ilist_nocpl = np.loadtxt("no_coupling/ilist_no_coupling{}_region{}.txt".format(year, region), dtype = int)
    elif region == 3:
        no_cpl_correlation = np.loadtxt("no_coupling/correlation_no_coupling{}_region{}.txt".format(year,region), dtype= np.float64)
        no_cpl_cell = np.loadtxt("no_coupling/cell_no_coupling{}_region{}.txt".format(year, region), dtype= np.float64)
        ilist_nocpl = np.loadtxt("no_coupling/ilist_no_coupling{}_region{}.txt".format(year, region), dtype = int)

elif int(year) == 2007:
    if region == 0:
        no_cpl_correlation = np.loadtxt("no_coupling/correlation_no_coupling{}_region{}.txt".format(year,region), dtype= np.float64)
        no_cpl_cell = np.loadtxt("no_coupling/cell_no_coupling{}_region{}.txt".format(year, region), dtype= np.float64)
        ilist_nocpl = np.loadtxt("no_coupling/ilist_no_coupling{}_region{}.txt".format(year, region), dtype = int)
    elif region == 1:
        no_cpl_correlation = np.loadtxt("no_coupling/correlation_no_coupling{}_region{}.txt".format(year,region), dtype= np.float64)
        no_cpl_cell = np.loadtxt("no_coupling/cell_no_coupling{}_region{}.txt".format(year, region), dtype= np.float64)
        ilist_nocpl = np.loadtxt("no_coupling/ilist_no_coupling{}_region{}.txt".format(year, region), dtype = int)
    elif region == 2:
        no_cpl_correlation = np.loadtxt("no_coupling/correlation_no_coupling{}_region{}.txt".format(year,region), dtype= np.float64)
        no_cpl_cell_400 = np.loadtxt("no_coupling/cell_no_coupling{}_region{}_400.txt".format(year, region), dtype= np.float64)
        no_cpl_cell_401 = np.loadtxt("no_coupling/cell_no_coupling{}_region{}_401.txt".format(year, region), dtype= np.float64)
        ilist_nocpl = np.loadtxt("no_coupling/ilist_no_coupling{}_region{}.txt".format(year, region), dtype = int)
    elif region == 3:
        no_cpl_correlation = np.loadtxt("no_coupling/correlation_no_coupling{}_region{}.txt".format(year,region), dtype= np.float64)
        no_cpl_cell = np.loadtxt("no_coupling/cell_no_coupling{}_region{}.txt".format(year, region), dtype= np.float64)
        ilist_nocpl = np.loadtxt("no_coupling/ilist_no_coupling{}_region{}.txt".format(year, region), dtype = int)
        
elif int(year) == 2010:
    if region == 2:
        no_cpl_correlation = np.loadtxt("no_coupling/correlation_no_coupling{}_region{}.txt".format(year,region), dtype= np.float64)
        no_cpl_cell_400 = np.loadtxt("no_coupling/cell_no_coupling{}_region{}_400.txt".format(year, region), dtype= np.float64)
        no_cpl_cell_420 = np.loadtxt("no_coupling/cell_no_coupling{}_region{}_420.txt".format(year, region), dtype= np.float64)
        no_cpl_cell_440 = np.loadtxt("no_coupling/cell_no_coupling{}_region{}_440.txt".format(year, region), dtype= np.float64)
        ilist_nocpl = np.loadtxt("no_coupling/ilist_no_coupling{}_region{}.txt".format(year, region), dtype = int)
    else:
        no_cpl_correlation = np.loadtxt("no_coupling/correlation_no_coupling{}_region{}.txt".format(year,region), dtype= np.float64)
        no_cpl_cell = np.loadtxt("no_coupling/cell_no_coupling{}_region{}.txt".format(year, region), dtype= np.float64)
        ilist_nocpl = np.loadtxt("no_coupling/ilist_no_coupling{}_region{}.txt".format(year, region), dtype = int)



#Plotting sequence for spatial variance

fig = plt.figure(figsize = (8,6))
ax1 = fig.add_subplot(111)
ax1.set_ylim(-1.0,1.8)

if int(year) == 2005:
    if region == 0:
        line1, =  ax1.plot(i_list_new, av_47, 'y', linestyle = "-", label = "Average state for cell 47 ")
        line2, = ax1.plot(ilist_nocpl, no_cpl_cell, 'y', linestyle = "--", label = "Average state for no_cpl 47")
    elif region ==1:
        line1, =  ax1.plot(i_list_new, av_172, 'y', linestyle = "-", label = "Average state for cell 172 ")
        line2, = ax1.plot(ilist_nocpl, no_cpl_cell, 'y', linestyle = "--", label = "Average state for no_cpl 172")
    elif region == 2:
        line1, = ax1.plot(i_list_new, av_400, 'y', label = "Average state for cell 400")
        line2, = ax1.plot(i_list_new, av_401, 'b', label = "Average state for cell 401")
        line3, = ax1.plot(i_list_new, av_420, 'r', label = "Average state for cell 420")
        line4, = ax1.plot(i_list_new, av_421, 'm', label = "Average state for cell 421")
        line5, = ax1.plot(i_list_new, av_440, 'c', label = "Average state for cell 440")
        line6, = ax1.plot(i_list_new, av_441, 'k', label = "Average state for cell 441")
        line7, = ax1.plot(ilist_nocpl, no_cpl_cell_400, 'y', linestyle = "--", label = "Average state for no_cpl 400")
        line8, = ax1.plot(ilist_nocpl, no_cpl_cell_401, 'b', linestyle = "--",label = "Average state for no_cpl 401")
        line9, = ax1.plot(ilist_nocpl, no_cpl_cell_420, 'r', linestyle = "--",label = "Average state for no_cpl 420")
        line10, = ax1.plot(ilist_nocpl, no_cpl_cell_421, 'm',linestyle = "--", label = "Average state for no_cpl 421")
        line11, = ax1.plot(ilist_nocpl, no_cpl_cell_440, 'c',linestyle = "--", label = "Average state for no_cpl 440")
        line12, = ax1.plot(ilist_nocpl, no_cpl_cell_441, 'k',linestyle = "--", label = "Average state for no_cpl 441")
    elif region == 3:
        line1, =  ax1.plot(i_list_new, av_538, 'y', linestyle = "-", label = "Average state for cell 538 ")
        line2, = ax1.plot(ilist_nocpl, no_cpl_cell, 'y', linestyle = "--", label = "Average state for no_cpl 538")
elif int(year) == 2007:
    if region == 0:
        line1, =  ax1.plot(i_list_new, av_30, 'y', linestyle = "-", label = "Average state for cell 30 ")
        line2, = ax1.plot(ilist_nocpl, no_cpl_cell, 'y', linestyle = "--", label = "Average state for no_cpl 30")
    elif region ==1:
        line1, =  ax1.plot(i_list_new, av_218, 'y', linestyle = "-", label = "Average state for cell 218 ")
        line2, = ax1.plot(ilist_nocpl, no_cpl_cell, 'y', linestyle = "--", label = "Average state for no_cpl 218")
    elif region == 2:
        line1, = ax1.plot(i_list_new, av_400, 'y', label = "Average state for cell 400")
        line2, = ax1.plot(i_list_new, av_401, 'b', linestyle = "--", label = "Average state for no_cpl 400")
        line3, = ax1.plot(ilist_nocpl, no_cpl_cell_400, 'y', linestyle = "--",label = "Average state for no_cpl 400")
        line4, = ax1.plot(ilist_nocpl, no_cpl_cell_401, 'b', linestyle = "--", label = "Average state for no_cpl 401")
    elif region == 3:
        line1, =  ax1.plot(i_list_new, av_540, 'y', linestyle = "-", label = "Average state for cell 540")
        line2, = ax1.plot(ilist_nocpl, no_cpl_cell, 'y', linestyle = "--", label = "Average state for no_cpl 540")
elif int(year) == 2010:
    if region == 0:
        line1, =  ax1.plot(i_list_new, av_30, 'y', linestyle = "-", label = "Average state for cell 30 ")
        line2, = ax1.plot(ilist_nocpl, no_cpl_cell, 'y', linestyle = "--", label = "Average state for no_cpl 30")
    elif region ==1:
        line1, =  ax1.plot(i_list_new, av_216, 'y', linestyle = "-", label = "Average state for cell 216 ")
        line2, = ax1.plot(ilist_nocpl, no_cpl_cell, 'y', linestyle = "--", label = "Average state for no_cpl 216")
    elif region == 2:
        line1, = ax1.plot(i_list_new, av_400, 'y', label = "Average state for cell 400")
        line2, = ax1.plot(i_list_new, av_420, 'r', label = "Average state for cell 420")
        line3, = ax1.plot(i_list_new, av_440, 'c', label = "Average state for cell 421")
        line4, = ax1.plot(ilist_nocpl, no_cpl_cell_400, 'y', linestyle = "--",label = "Average state for no_cpl 400")
        line5, = ax1.plot(ilist_nocpl, no_cpl_cell_420, 'r', linestyle = "--", label = "Average state for no_cpl 420")
        line6, = ax1.plot(ilist_nocpl, no_cpl_cell_440, 'c', linestyle = "--",label = "Average state for no_cpl 440")
    elif region == 3:
        line1, =  ax1.plot(i_list_new, av_538, 'y', linestyle = "-", label = "Average state for cell 538")
        line2, = ax1.plot(ilist_nocpl, no_cpl_cell, 'y', linestyle = "--", label = "Average state for no_cpl 538")
else:
    pass

ax1.set_xlabel('Timeseries')
ax1.set_ylabel('Average state for tipped cells', color = 'black')
ax1.tick_params(axis='x', labelsize=8)
ax1.tick_params(axis='y', labelsize = 8)

ax2 = ax1.twinx()

if int(year) == 2005:
    if region == 2:
        line13, = ax2.plot(i_list_new, correlation, 'g', label = "Spatial correlation")
        line14, = ax2.plot(ilist_nocpl, no_cpl_correlation, 'g', linestyle = "--", label = "Spatial correlation no_cpl")
        plt.legend((line1, line2, line3, line4, line5, line6, line7, line8, line9, line10, line11, line12, line13, line14), 
        ('Average state for cell 400', 'Average state for cell 401', 'Average state for cell 420', 
                                'Average state for cell 421', 'Average state for cell 440', 'Average state for cell 441', 
                                'Average state for no_cpl 400', 'Average state for no_cpl 401', 'Average state for no_cpl 420', 
                                'Average state for no_cpl 421', 'Average state for no_cpl 440', 'Average state for no_cpl 441',
                                'Spatial correlation', 'Spatial correlation n_cpl'), prop={'size': 8}, loc='upper left')
    else:
        line3, = ax2.plot(i_list_new, correlation, 'g', label = "Spatial correlation")
        line4, = ax2.plot(ilist_nocpl, no_cpl_correlation, 'g', linestyle = "--", label = "Spatial correlation no_cpl") 
        plt.legend((line1, line2, line3, line4), 
        ('Average state for cell ', 'Average state for cell no_cpl', 'Spatial correlation', 'Spatial correlation no_cpl'), prop={'size': 8}, loc='upper left')
elif int(year) == 2007:
    if region == 2:
        line13, = ax2.plot(i_list_new, correlation, 'g', label = "Spatial correlation")
        line14, = ax2.plot(ilist_nocpl, no_cpl_correlation, 'g', linestyle = "--", label = "Spatial correlation no_cpl")
        plt.legend((line1, line2, line3, line4, line13, line14), 
        ('Average state for cell 400', 'Average state for cell 401','Average state for no_cpl 400', 'Average state for no_cpl 401',
                                'Spatial correlation', 'Spatial correlation n_cpl'), prop={'size': 8}, loc='upper left')
    else:
        line3, = ax2.plot(i_list_new, correlation, 'g', label = "Spatial correlation")
        line4, = ax2.plot(ilist_nocpl, no_cpl_correlation, 'g', linestyle = "--", label = "Spatial correlation no_cpl") 
        plt.legend((line1, line2, line3, line4), 
        ('Average state for cell ', 'Average state for cell no_cpl', 'Spatial correlation', 'Spatial correlation no_cpl'), prop={'size': 8}, loc='upper left')
elif int(year) == 2010:
    if region == 2:
        line13, = ax2.plot(i_list_new, correlation, 'g', label = "Spatial correlation")
        line14, = ax2.plot(ilist_nocpl, no_cpl_correlation, 'g', linestyle = "--", label = "Spatial correlation no_cpl")
        plt.legend((line1, line2, line3, line4, line5, line6, line13, line14), 
        ('Average state for cell 400', 'Average state for cell 420','Average state for cell 440','Average state for no_cpl 400', 'Average state for no_cpl 420',
        'Average state for no_cpl 440', 'Spatial correlation', 'Spatial correlation n_cpl'), prop={'size': 8}, loc='upper left')
    else:
        line3, = ax2.plot(i_list_new, correlation, 'g', label = "Spatial correlation")
        line4, = ax2.plot(ilist_nocpl, no_cpl_correlation, 'g', linestyle = "--", label = "Spatial correlation no_cpl") 
        plt.legend((line1, line2, line3, line4), 
        ('Average state for cell', 'Average state for cell no_cpl', 'Spatial correlation', 'Spatial correlation no_cpl'), prop={'size': 8}, loc='upper left')


ax2.set_ylabel('Spatial Correlation', color = 'g' )
ax2.tick_params(axis='y', labelsize = 8)
axes = plt.gca()
axes.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
ax1.xaxis.label.set_size(10)
ax1.yaxis.label.set_size(10)
ax2.yaxis.label.set_size(10)

plt.axvline(x=i_list_new[-1]) 
plt.axvline(x=ilist_nocpl[-1], linestyle = "--")      

#plt.title("Varying c-values for cusps 468, 487 and 505, selected network upper right (0.01*60 rate)", fontsize=10)
plt.title("Spatial Correlation plot approaching drought scenario of year {} with {}".format(year_type, id), fontsize=10)
plt.tight_layout()

#if no_cpl_dummy == True:
#    fig.savefig("no_coupling/spat_var_unstable_amaz_{}_{}_adaptsample{}_adaptsigma{}_{}_{}_noise{}_std1.png".format(resolution_type, 
#        year_type, str(start_file).zfill(3), int(np.around(100*adapt)), id, float(rain_fact), noise), dpi=200)
#else:
if no_cpl_dummy == False:
    fig.savefig("corr_unstable_amaz_{}_{}_adaptsample{}_adaptsigma{}_field_number{}_{}_noise{}_std1.png".format(resolution_type, 
        year_type, str(start_file).zfill(3), int(np.around(100*adapt)), id, float(rain_fact), noise), dpi=200)







'''
#Plotting sequence for correlation
fig = plt.figure(figsize = (8,6))
ax1 = fig.add_subplot(111)
ax1.set_ylim(-1.0,1.4)
line1, = ax1.plot(i_list, av_505, 'b', label = "Average state for cusp 505")
line2, = ax1.plot(i_list, av_468, 'y', label = "Average state for cusp 468")
# line3, = ax1.plot(i_list, av_487, 'r', label = "Average state for cusp 487")
ax1.set_xlabel('Timeseries')
ax1.set_ylabel('Average state for forced cusps', color = 'black')
ax1.tick_params(axis='x', labelsize=8)
ax1.tick_params(axis='y', labelsize = 8)

ax2 = ax1.twinx()
line3, = ax2.plot(i_list, correlation, 'g', label = "Spatial correlation")
ax2.set_ylabel('Spatial correlation', color = 'g' )
ax2.tick_params(axis='y', labelsize = 8)
axes = plt.gca()
axes.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
ax1.xaxis.label.set_size(10)
ax1.yaxis.label.set_size(10)
ax2.yaxis.label.set_size(10)
plt.legend((line1, line2, line3), ('Average state for cusp 505', 'Average state for cusp 468', 'Spatial correlation'), prop={'size': 8}, loc='upper right')
plt.axvline(x=c_corr[0])

#plt.title("Varying c-values for cusps 468, 487 and 505, selected network upper right (0.01*60 rate)", fontsize=10)
plt.title("Spatial correlation plot approaching drought of year {} scenario with {}".format(year_type, id), fontsize=10)
plt.tight_layout()
fig.savefig("cor_unstable_amaz_{}_{}_adaptsample{}_adaptsigma{}_field_number{}_{}_noise{}_std1.png".format(resolution_type, 
        year_type, str(start_file).zfill(3), int(np.around(100*adapt)), id, float(rain_fact), noise), dpi=200)
'''