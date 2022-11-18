#use python3 r_crit_unstable_amazon.py 0 2004 0 1.0 1 
#Chage 2004 to the year you want to examine



# %%


import time
import numpy as np
import networkx as nx
import glob
import re
import os
from scipy import stats

import csv
from netCDF4 import Dataset

#plotting imports
import matplotlib


import matplotlib.pyplot as plt
import matplotlib.colors as colors
import cartopy.crs as ccrs
import cartopy.feature as cfeature

import seaborn as sns

#from r_crit_unstable_amazon_corr import tip_cor
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
# from evolve import evolve, NoEquilibrium
from evolve_sde import evolve, NoEquilibrium
from tipping_element import cusp
from coupling import linear_coupling
from functions_amazon import global_functions



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
# data_eval = data_crit

#evaluated drought (or not drought) year for the roughest periods from 2003-2014
hydro_1 = np.sort(np.array(glob.glob("more_data/monthly_data/correct_data/both_1deg/*{}*.nc".format(int(year)-1))))[9:]
hydro_2 = np.sort(np.array(glob.glob("more_data/monthly_data/correct_data/both_1deg/*{}*.nc".format(int(year)))))[:9]
data_eval = np.concatenate((hydro_1, hydro_2))

# cells = list(range(446, 454)) + list(range(466, 473)) + list(range(485,492)) + list(range(504, 510)) + list(range(521,525)) + list(range(534, 537)) +list(range(544, 547))
# cells = list(range(466, 473)) + list(range(485,492)) + list(range(504, 510)) + list(range(521,525)) + list(range(534, 537))
# c_cells = [468, 487, 505]

#Load neighbourlist to compute correlation between cells from r_crit_unstable_amazon.py

#c_values = np.loadtxt("./jobs/results/noise/final/c_end_values.txt", delimiter=" ",usecols=np.arange(0, n_cols))
c_end = np.loadtxt("./jobs/results/noise/final/c_end_values_2005.txt", usecols = (1), dtype= np.float64)
#Get all negavtive c-values converted to 0
c_end[c_end < 0] = 0
#print(np.sort(c_end))

c_begin = np.loadtxt("./jobs/results/noise/final/c_begin_values.txt", usecols = 1, dtype = np.float64)
c_begin[c_begin == -0.0] = 0
#c_values = np.loadtxt("./jobs/results/noise/final/c_end_values.txt", dtype=dtype)
#dc = (c_end/200)


#Tipping function
def tip( net , initial_state ):
    
    ev = evolve( net , initial_state )

    tolerance = 0.1

    if not ev.is_equilibrium(tolerance):
        #This network is not in equilibrium since it[0] (= moisture transport) is not effectless at the current state of the network,
      print("Warning: Initial state is not a fixed point of the system")
    elif not ev.is_stable():
        print("Warning: Initial state is not a stable point of the system")

    t_step = 0.1
    realtime_break = 100 #originally 30000, works with 200

    #ev = evolve (net, ev.get_timeseries()[1][-1])
    #ev.integrate(t_step, realtime_break, ev.get_timeseries()[1][-1], sigma=sigma)
    sigmas = np.array([0.01] * np.ones(net.number_of_nodes()))
    alphas = 2 * np.ones(net.number_of_nodes())
    ev.integrate(t_step, realtime_break, sigmas = sigmas, noise = "normal", alphas = alphas, seed = None)
    #cusp_all = ev.get_timeseries()[1][:,:].T

    # Introduce same noise for all the elements of the network
    # noise = 0.01 # used the same as in my_r_crit
    # sigma = np.diag([1] * net.number_of_nodes()) * noise
   
    '''
    # Code to integrate from Status 0 to status drought and examine the difference between c-lists
    c_lis = []
    for i in range(0, 10000):    #Set 10000 as it stops anyway with exit()
        for x in range(0, net.number_of_nodes()):
            if net.nodes[x]['data'].get_par()['c'] <= c_end[x]:
                c = net.nodes[x]['data'].get_par()['c']
                # print(f"C-Values are,", c)
                net.set_param(x, 'c', c+dc[x])
                if i == 199:
                    c_lis.append(c)
                else:
                    pass
            else:
                print(f"c_list is", c_lis)
                print(f"Lenght of c_lis is,", len(c_lis))
                diff = [x1 - x2 for (x1, x2) in zip(list(c_end), c_lis)]
                print(f"Scenario of hydro has been reached")
                print(f"Maximum difference between at dc_500 c_lists are", max(diff))
                print(f"The statistical t-test result for both c-series is:", stats.ttest_ind(c_end, c_lis))
                # print(c_end[i])
                print(f"Value of i where Scenario of hydro has been reached", i)
                print(f"Value for node 0 at beginning of simulation is:", net.nodes[0]['data'].get_par()['c'])
                print(f"Value for cell 1 at the beginning of simulation is,", net.nodes[1]['data'].get_par()['c'])
                exit()
        

        ev = evolve (net, ev.get_timeseries()[1][-1])
        sigmas = np.array([0.01] * np.ones(net.number_of_nodes()))
        alphas = 2 * np.ones(net.number_of_nodes())
        ev.integrate(t_step, realtime_break, sigmas = sigmas, noise = "normal", alphas = alphas, seed = None)

        print(f"Value for node 0 at end of simulation is:", net.nodes[0]['data'].get_par()['c'])
        print(f"Value for cell 1 at the end of simulation is,", net.nodes[1]['data'].get_par()['c'])
        print(f"Value of c for first tipped is:", c if net.get_tip_states(ev.get_timeseries()[1][-1])[:].any() != False else "None")
    '''
           

    
    # dc = 0.005
    # dc = np.arange(0, 0.7, 0.01)
    # dc = np.array([0.01*60])  # with dc = np.array([0.01] * 20)  and realtime_break = 200 works for 0.85 tipping cascade
    # for i in range(0, len(dc)):
        #c = net.nodes[505]['data'].get_par()['c'] 
        #net.set_param(505, 'c', c+dc[i])

        #for cell in c_cells:
        #    c = net.nodes[cell]['data'].get_par()['c'] 
        #    net.set_param(cell, 'c', c+dc[i])
        # c_lis.append(c)

        #print(f"Value for c of cell 505:", c)

    # ev = evolve (net, ev.get_timeseries()[1][-1])
    # ev.integrate(t_step, realtime_break, ev.get_timeseries()[1][-1], sigma=sigma)
    # sigmas = np.array([0.01] * np.ones(net.number_of_nodes()))
    # alphas = 2 * np.ones(net.number_of_nodes())
    # ev.integrate(t_step, realtime_break, sigmas = sigmas, noise = "normal", alphas = alphas, seed = None)
    # cusp_all = ev.get_timeseries()[1][:,:].T
    
    conv_time = ev.get_timeseries()[0][-1] - ev.get_timeseries()[0][0]
    return conv_time, net.get_number_tipped(ev.get_timeseries()[1][-1,:]), net.get_tip_states(ev.get_timeseries()[1][-1])[:]
    # return conv_time, num_tipped, tipped



###MAIN - PREPARATION###
#need changing variables from file names
dataset = data_crit[0]
net_data = Dataset(dataset)

#latlon values
lat = net_data.variables["lat"][:]
lon = net_data.variables["lon"][:]

resolution_type = "1deg"
year_type = year


#Network is created using the monthly data, the critical mcwd, coupling switch and the rain factor

net = generate_network(data_crit, data_crit_std, data_eval, no_cpl_dummy, rain_fact, adapt_fact)

###MAIN - PREPARATION###
output = []

init_state = np.zeros(net.number_of_nodes())
init_state.fill(-1) #initial state should be -1 instead of 0 everywhere; this means amazon is covered with veg.

#Without the source node tipped
info = tip(net, init_state)
conv_time = info[0]
casc_size = info[1] 
unstable_amaz = info[2]
#id = f""
noise = f"normal"

# nochmal os.chdir neu definieren um outputs/errors abzuspeichern
os.chdir("/p/projects/dominoes/lena")

'''
if no_cpl_dummy == True:
    np.savetxt("r_crit/no_coupling/unstable_amaz_{}_{}_adaptsample{}_adaptsigma{}_field.txt".format(resolution_type, 
        year_type, str(start_file).zfill(3), int(np.around(100*adapt))), unstable_amaz)
    np.savetxt("r_crit/no_coupling/unstable_amaz_{}_{}_adaptsample{}_adaptsigma{}_total.txt".format(resolution_type, 
        year_type, str(start_file).zfill(3), int(np.around(100*adapt))), [conv_time, casc_size])
else:
    np.savetxt("unstable_amaz_{}_{}_adaptsample{}_adaptsigma{}_field_number{}_{}_100_noise{}.txt".format(resolution_type, 
        year_type, str(start_file).zfill(3), int(np.around(100*adapt)), id, float(rain_fact), noise), unstable_amaz)
    np.savetxt("unstable_amaz_{}_{}_adaptsample{}_adaptsigma{}_total_number{}_{}_100_noise{}.txt".format(resolution_type, 
        year_type, str(start_file).zfill(3), int(np.around(100*adapt)), id, float(rain_fact), noise), [conv_time, casc_size])
'''

#plotting procedure
print("Plotting sequence")
tuples = [(lat[idx],lon[idx]) for idx in range(lat.size)]

'''
#Getting the cells belonging to Regions defined by IPCC 6

#NES = []
neighbours = []
tuples_NWS = [(lat, lon) for lat in range(-17, +11) for lon in range(-79, -71)]
#print(np.sort(tuples_NWS))
tuples_NSA = [(lat, lon) for lat in range(-7, +11) for lon in range(-71, -50)]
tuples_SAM = [(lat, lon) for lat in range(-17, -7) for lon in range(-71, -50)]
tuples_NES = [(lat, lon) for lat in range(-17, 2) for lon in range(-50, -44)]


for (x,y) in tuples_NES:
    if (x,y) in tuples: 
        x_1, y_1, x1, y1 = x - 1, y - 1, x + 1, y + 1

        if (x_1, y) in tuples and (x_1, y) in tuples_NES:  # P2
            #if tuples.index((x_1,y)) in list(NES):
                #nghbr = [tuples.index((x_1,y)), tuples.index((x_1,y1)), tuples.index((x,y1)), tuples.index((x1,y1)),     # P2,P3,P4,P5
                #      tuples.index((x1,y)), tuples.index((x1,y_1)), tuples.index((x, y_1)), tuples.index((x_1,y_1))]    # P6,P7,P8,P9
                nghbr = (tuples.index((x,y)), tuples.index((x_1, y)))
                neighbours.append(nghbr)
        else:
            pass

        if (x_1,y_1) in tuples and (x_1, y_1) in tuples_NES: # P3
            #if tuples.index((x_1,y_1)) in list(NES):
            nghbr = (tuples.index((x,y)),tuples.index((x_1,y_1)))
            neighbours.append(nghbr)
        else:
            pass
        
        if (x,y1) in tuples and (x, y1) in tuples_NES:   # P4
            #if tuples.index((x, y1)) in list(NES):
            nghbr = (tuples.index((x,y)), tuples.index((x,y1)))
            neighbours.append(nghbr)
        else:
            pass

        if (x1,y1) in tuples and (x1, y1) in tuples_NES: #P5
            #if tuples.index((x1, y1)) in list(NES):
            nghbr = (tuples.index((x,y)),tuples.index((x1,y1)))
            neighbours.append(nghbr)
        else:
            pass

        if (x1,y) in tuples and (x1, y) in tuples_NES: #P6
            #if tuples.index((x1, y)) in list(NES):
            nghbr = (tuples.index((x,y)),tuples.index((x1,y)))
            neighbours.append(nghbr)
        else:
            pass
        
        if (x1,y_1) in tuples and (x1, y_1) in tuples_NES: #P7
            #if tuples.index((x1, y_1)) in list(NES):
            nghbr = (tuples.index((x,y)),tuples.index((x1,y_1)))
            neighbours.append(nghbr)
        else:
            pass

        if (x,y_1) in tuples and (x, y_1) in tuples_NES: #P8
            #if tuples.index((x, y_1)) in list(NES):
            nghbr = (tuples.index((x,y)),tuples.index((x,y_1)))
            neighbours.append(nghbr)
        else:
            pass

        if (x_1,y1) in tuples and (x_1, y1) in tuples_NES: #P9
            #if tuples.index((x_1, y_1)) in list(NES):
            nghbr = (tuples.index((x,y)),tuples.index((x_1,y1)))
            neighbours.append(nghbr)
        else:
            pass

                

print(f"Sorted list of nieghbours is:", sorted(neighbours))
np.savetxt("./jobs/results/noise/neighbourslist_NES.txt", neighbours, fmt = '%d')
'''

'''
for (x,y) in tuples_NES:
    if (x,y) in tuples:
        item = (tuples.index((x,y)))
        NES.append(item)
    else:
        pass

np.savetxt("NES_cells.txt", np.sort(NES), fmt = '%d')
'''


lat = np.unique(lat)
lon = np.unique(lon)
lat = np.append(lat, lat[-1]+lat[-1]-lat[-2]) #why do we need to append lat[-1]+lat[-1]-lat[-2]???
lon = np.append(lon, lon[-1]+lon[-1]-lon[-2])
vals = np.empty((lat.size,lon.size)) #Vals is True or False depending on if the cells tipped
vals[:,:] = np.nan


for idx,x in enumerate(lat):
    for idy,y in enumerate(lon):
        if (x,y) in tuples:
        #Get all cells values if tipped or not in varible p
            p = unstable_amaz[tuples.index((x,y))]
            # vals[idx,idy] = p
            c_value = c_begin[tuples.index((x,y))]
            vals[idx,idy] = c_value
            '''
            #Get neighbourslist for further correlation analysis; compute neighbours for each cellin clockwise direction (8 pairs max. as tuples)
            x_1, y_1, x1, y1 = x - 1, y - 1, x + 1, y + 1
            
            if (x_1, y) in tuples:  # P2
                #nghbr = [tuples.index((x_1,y)), tuples.index((x_1,y1)), tuples.index((x,y1)), tuples.index((x1,y1)),     # P2,P3,P4,P5
                #      tuples.index((x1,y)), tuples.index((x1,y_1)), tuples.index((x, y_1)), tuples.index((x_1,y_1))]    # P6,P7,P8,P9
                nghbr = (tuples.index((x,y)), tuples.index((x_1, y)))
                neighbours.append(nghbr)
            else:
                pass

            if (x_1,y_1) in tuples: # P3
                nghbr = (tuples.index((x,y)),tuples.index((x_1,y_1)))
                neighbours.append(nghbr)
            else:
                pass
            
            if (x,y1) in tuples:   # P4
                nghbr = (tuples.index((x,y)), tuples.index((x,y1)))
                neighbours.append(nghbr)
            else:
                pass

            if (x1,y1) in tuples: #P5
                nghbr = (tuples.index((x,y)),tuples.index((x1,y1)))
                neighbours.append(nghbr)
            else:
                pass

            if (x1,y) in tuples: #P6
                nghbr = (tuples.index((x,y)),tuples.index((x1,y)))
                neighbours.append(nghbr)
            else:
                pass
            
            if (x1,y_1) in tuples: #P7
                nghbr = (tuples.index((x,y)),tuples.index((x1,y_1)))
                neighbours.append(nghbr)
            else:
                pass

            if (x,y_1) in tuples: #P8
                nghbr = (tuples.index((x,y)),tuples.index((x,y_1)))
                neighbours.append(nghbr)
            else:
                pass

            if (x_1,y1) in tuples: #P9
                nghbr = (tuples.index((x,y)),tuples.index((x_1,y1)))
                neighbours.append(nghbr)
            else:
                pass
   
            
print(sorted(neighbours))
np.savetxt("./jobs/results/noise/neighbourslist.txt", neighbours, fmt = '%d')
'''

os.chdir("/p/projects/dominoes/lena/jobs/results/noise/final")

plt.rc('text', usetex=False)
plt.rc('font', family='serif', size=25)

plt.figure(figsize=(15,10))

ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_extent([275, 320, -22, 15], crs=ccrs.PlateCarree())
ax.add_feature(cfeature.COASTLINE, linewidth=1)
ax.coastlines('50m')
#cmap = plt.get_cmap('turbo')
#cmap = plt.get_cmap('ocean')

#cmap = matplotlib.cm.ocean(np.linspace(0,1,20))
#cmap = matplotlib.colors.ListedColormap(cmap[0:, :-1])
cmap = matplotlib.colors.LinearSegmentedColormap.from_list('summer', ['#1B5E20', '#1B5E20']) # '#D4E157' ,'#FFEE58','#FF8F00'])

#Plot c-values instead of vals (tipped) values

plt.pcolor(lon-(lon[-1]-lon[-2]) / 2, lat-(lat[-1]-lat[-2]) / 2, vals, cmap=cmap)

#plt.pcolor(lon-(lon[-1]-lon[-2]) / 2, lat-(lat[-1]-lat[-2]) / 2, vals, cmap=cmap)
#nx.draw_networkx(net,pos, edge_color='black', node_size=0, with_labels=False)
cbar = plt.colorbar(label='Critical parameter c')
tick_font_size = 12
cbar.ax.tick_params(labelsize=tick_font_size)

if no_cpl_dummy == True:
    plt.savefig("r_crit/no_coupling/unstable_amaz_{}_{}_adaptsample{}_adaptsigma{}_notwhite2.png".format(resolution_type, year_type, 
        str(start_file).zfill(3), int(np.around(100*adapt))), bbox_inches='tight')
    plt.savefig("r_crit/no_coupling/unstable_amaz_{}_{}_adaptsample{}_adaptsigma{}_notwhite2.pdf".format(resolution_type, year_type, 
        str(start_file).zfill(3), int(np.around(100*adapt))), bbox_inches='tight')
else:
    plt.savefig("c_begin_unstable_amaz_{}_{}.png".format(resolution_type, year_type), bbox_inches='tight')
    plt.savefig("c_begin_unstable_amaz_{}_{}.pdf".format(resolution_type, year_type), bbox_inches='tight')

#plt.show()
plt.clf()
plt.close()


print("Finish")
# %%
