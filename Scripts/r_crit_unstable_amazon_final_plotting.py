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
from scipy.stats import kendalltau

"This code can be used to calculate the correlation between neighbouring cells (and plot it against average state of cell) for specially selected cells when increasing c_value for one/or more cells"
"The spatial correlation index was defined as the two-point correlation for all pairs of cells separated by distance 1, using the Moran’s coefficient (Legendre)"

#sys_var = np.array(sys.argv[2:])
#year = sys_var[0]
#no_cpl_dummy = int(sys_var[1])
#adapt = float(sys_var[2])       #Range of adaptability in multiples of the standard deviation; higher adapt_fact means higher adaptability
#start_file = sys_var[3]

#year = 2004
#no_cpl_dummy = 1
adapt = 1.0
start_file = 1
year = 2010
region = 0
no_cpl_dummy = 0
#coupling = 6.0


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
#c_values = np.loadtxt("./jobs/results/noise/final/c_end_values.txt", delimiter=" ",usecols=np.arange(1, n_cols))
#c_end = np.loadtxt("./jobs/results/noise/final/c_end_values.txt", usecols = (1), dtype= np.float64)
#Get all negavtive c-values converted to 0
#c_end[c_end < 0] = 0

c_begin = np.loadtxt("./jobs/results/noise/final/c_begin_values.txt", usecols = 1, dtype = np.float64)

#c_values = np.loadtxt("./jobs/results/noise/final/c_end_values.txt", dtype=dtype)

# Choose betwen NWS(0), NSA(1), SAM(2), NES(3)


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
    cells = list(range(0, 567))


c_end[ c_end < 0] = 0
#Try out not including the c>2 values for 2005
#c_end[c_end >= 2.0] = 0
t_step = 0.1
realtime_break = 100 #originally 30000 and works with 200 (see r_crt_unstable_amazon.py)
timesteps = (realtime_break/t_step)
dc = (c_end/timesteps)
print(f"Cells are", cells)
print(f"c_end values are", c_end)
print(f"The values for dc are:", dc)


os.chdir("/p/projects/dominoes/lena")

'''
#Load no_coupling and coupling variables for given year and region
if no_cpl_dummy == True:
    ilist = np.loadtxt("./jobs/results/noise/final/{}/no_coupling/i_list_{}_region{}.txt".format(year, year, region), dtype = int)
    print(f"Shape of ilist is", np.shape(ilist))
    all_states = np.loadtxt("./jobs/results/noise/final/{}/no_coupling/states_cusps_{}_region{}.txt".format(year, year, region), dtype = np.float64)
else:
    ilist = np.loadtxt("./jobs/results/noise/final/{}/i_list_{}_region{}.txt".format(year, year, region), dtype = int)
    all_states = np.loadtxt("./jobs/results/noise/final/{}/states_cusps_{}_region{}.txt".format(year, year, region), dtype = np.float64)

#####To get state values for all years, before and after tip for coupling and non-coupling############
dataset = data_crit[0]
net_data = Dataset(dataset)

regions = [0,1,2,3]

for region in regions:

    os.chdir("/p/projects/dominoes/lena")
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
    
    if no_cpl_dummy == True:
        os.chdir("/p/projects/dominoes/lena/jobs/results/noise/final/{}/no_coupling".format(year))
        states = np.loadtxt("states_cusps_{}_region{}.txt".format(year, region), dtype = np.float64)
    else:
        os.chdir("/p/projects/dominoes/lena/jobs/results/noise/final/{}".format(year))
        states = np.loadtxt("states_cusps_{}_region{}.txt".format(year, region), dtype = np.float64)
    
    #for i in np.arange(0, ilist[-1]+1, 100):
    num_cols = states.shape[1]
    cols_list = range(0, num_cols)

    for i in cols_list:
    #if i == ilist[20]: #i == ilist[-2] or i == ilist[-1]:  #i % 100 == 0 or  #For 2005 i % 500, for 2007/2010 i %100
            if (i == cols_list[int(len(cols_list)/2)] or i == cols_list[-2] or i == cols_list[-1]):
                #latlon values
                lat = net_data.variables["lat"][:]
                lon = net_data.variables["lon"][:]
                #c = net_data.variables["c"][:]

                resolution_type = "1deg"
                year_type = year

                #latlon values
                        
                tuples = [(lat[idx],lon[idx]) for idx in range(lat.size)]

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
                            #print(f"Tuples index is.,", tuples.index((x,y)))
                            cell_index = list(cells).index(tuples.index((x,y)))
                            #print(f"Cell index is:,", cell_index)
                            #print(f"Last column of cval is", cval[:, 93])
                            
                            #c_value = net.nodes[cell_index]['data'].get_par()['c']
                            #print(f"(C-value, cell_index is", (c_value, cell_index))

                            #Get all cells values if tipped or not in varible p
                            #p = net.get_tip_states(ev.get_timeseries()[1][-1])[:][tuples.index((x,y))]
                            
                            #Get also intermediate values for cells
                            if no_cpl_dummy == True:
                                #i_new = int(i/100)
                                #p = all_states_nocpl[cell_index, i_new]
                                p = states[cell_index, i]
                                #print(f"p-value of tuple is", p)
                                #p = cval[cell_index, i_new]
                                vals[idx, idy] = p
                            else:
                                #i_new = int(i/100)
                                #p = all_states[cell_index, i_new]
                                #p = cval[cell_index, i_new]
                                p = states[cell_index, i]
                                vals[idx,idy] = p
                        else:
                            pass


                plt.rc('text', usetex=False)
                plt.rc('font', family='serif', size=25)

                plt.figure(figsize=(15,10))

                ax = plt.axes(projection=ccrs.PlateCarree())
                ax.set_extent([275, 320, -22, 15], crs=ccrs.PlateCarree())
                ax.add_feature(cfeature.COASTLINE, linewidth=1)
                ax.coastlines('50m')
                # cmap = plt.get_cmap('turbo')
                # cmap = plt.get_cmap('summer', 20)
                # cmap = matplotlib.cm.ocean(np.linspace(0,1,20))
                # cmap = matplotlib.cm.viridis(np.linspace(0,1,20)) #Workss but doesnt look that good
                # cmap = matplotlib.cm.gist_earth(np.linspace(0,1,20))
                # cmap = matplotlib.cm.summer(np.linspace(0,1,40))
                # cmap = matplotlib.colors.ListedColormap(cmap[2:15])  #Works for viridis [2:20]
                # cmap = matplotlib.colors.ListedColormap(cmap[9:19])  # For gist_earth
                # cmap = matplotlib.colors.ListedColormap(cmap[2:20])
                # cmap = matplotlib.colors.ListedColormap(["darkgreen", "forestgreen","limegreen", "lawngreen", "greenyellow", "darkkhaki", "palegoldenrod", "khaki", "goldenrod", "gold", "yellow"])
                cmap = matplotlib.colors.LinearSegmentedColormap.from_list('summer', ['#1B5E20','#D4E157','#FFEE58','#FF8F00'])

                plt.pcolor(lon-(lon[-1]-lon[-2]) / 2, lat-(lat[-1]-lat[-2]) / 2, vals, cmap=cmap)
                #nx.draw_networkx(net,pos, edge_color='black', node_size=0, with_labels=False)
                # cbar = plt.colorbar(label='Critical thresholds')

                # cbar = plt.colorbar(label='State values')
                cmap.set_over('red')
                plt.clim(-1.0, -0.50)
                #plt.clim(0, np.sqrt(4/27))
                if no_cpl_dummy == True:
                    if i == cols_list[-2]:
                        plt.savefig("beforetip_region{}_unstableamazon{:04n}".format(region, i), bbox_inches='tight')
                    elif i == cols_list[-1]:
                        plt.savefig("aftertip_region{}_unstableamazon{:04n}".format(region, i), bbox_inches='tight')
                    else:
                        plt.savefig("middle_region{}_unstableamazon{:04n}".format(region, i), bbox_inches='tight')
                else:
                    if i == cols_list[-2]:
                        plt.savefig("coupling/beforetip_region{}_unstableamazon{:04n}".format(region, i), bbox_inches='tight')
                    elif i == cols_list[-1]:
                        plt.savefig("coupling/aftertip_region{}_unstableamazon{:04n}".format(region, i), bbox_inches='tight')
                    else:
                        plt.savefig("coupling/middle_uregion{}_unstableamazon{:04n}".format(region, i), bbox_inches='tight') 

                plt.clf()
                plt.close() 

'''

###MAIN - PREPARATION###

###To get plots of different dc values#########
#need changing variables from file names
dataset = data_crit[0]
net_data = Dataset(dataset)

#latlon values
lat = net_data.variables["lat"][:]
lon = net_data.variables["lon"][:]
#c = net_data.variables["c"][:]

resolution_type = "1deg"
year_type = year

#latlon values
         
tuples = [(lat[idx],lon[idx]) for idx in range(lat.size)]

lat = np.unique(lat)
#print(f"Latitude values are", lat)
lon = np.unique(lon)
lat = np.append(lat, lat[-1]+lat[-1]-lat[-2]) 
lon = np.append(lon, lon[-1]+lon[-1]-lon[-2])

vals = np.empty((lat.size,lon.size)) #Vals is True or False depending on if the cells tipped
vals[:,:] = np.nan

#regions = [0,1,2,3]
regions = [4]

for region in regions:
    os.chdir("/p/projects/dominoes/lena")

    dataset = data_crit[0]
    net_data = Dataset(dataset)

    #latlon values
    lat = net_data.variables["lat"][:]
    lon = net_data.variables["lon"][:]
    #c = net_data.variables["c"][:]

    resolution_type = "1deg"
    year_type = year

    #latlon values
            
    tuples = [(lat[idx],lon[idx]) for idx in range(lat.size)]

    lat = np.unique(lat)
    #print(f"Latitude values are", lat)
    lon = np.unique(lon)
    lat = np.append(lat, lat[-1]+lat[-1]-lat[-2]) 
    lon = np.append(lon, lon[-1]+lon[-1]-lon[-2])

    vals = np.empty((lat.size,lon.size)) #Vals is True or False depending on if the cells tipped
    vals[:,:] = np.nan

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
        cells = list(range(0, 567))

    c_end[ c_end < 0] = 0
    #Try out not including the c>2 values for 2005
    #c_end[c_end >= 2.0] = 0
    t_step = 0.1
    realtime_break = 100 #originally 30000 and works with 200 (see r_crt_unstable_amazon.py)
    timesteps = (realtime_break/t_step)
    dc = (c_end/timesteps)
    #print(f"Cells are", cells)
    #print(f"c_end values are", c_end)
    #print(f"The values for dc are:", dc)

    if no_cpl_dummy == True:
        os.chdir("/p/projects/dominoes/lena/jobs/results/noise/final/{}/no_coupling".format(year))
        states = np.loadtxt("states_cusps_{}_region{}.txt".format(year, region), dtype = np.float64)
    else:
        os.chdir("/p/projects/dominoes/lena/jobs/results/noise/final/{}".format(year))
        states = np.loadtxt("states_cusps_{}_region{}.txt".format(year, region), dtype = np.float64)
    

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
        tuples_region = tuples
        lat = lat
        lon = lon

    tuple_list = []
    for idx,x in enumerate(lat):
        for idy,y in enumerate(lon):
            if (x,y) in tuples and (x,y) in tuples_region: #and (x,y) in tuples_region:
                tuple_list.append(tuples.index((x,y)))
                #print(f"(x,y) are", ((x,y)))
                #print(f"Tuples index is.,", tuples.index((x,y)))
                cell_index = list(cells).index(tuples.index((x,y)))
                #print(f"Cell index is:,", cell_index)
                #print(f"Last column of cval is", cval[:, 93])
                
                #c_value = net.nodes[cell_index]['data'].get_par()['c']
                #print(f"(C-value, cell_index is", (c_value, cell_index))

                #Get all cells values if tipped or not in varible p
                #p = net.get_tip_states(ev.get_timeseries()[1][-1])[:][tuples.index((x,y))]
                states = -1* np.ones((567, 1))
                states[:284, :] = -0.5
                #states[::2, :] = -0.5
                np.random.shuffle(states)
                #Get also intermediate values for cells
                if no_cpl_dummy == True:
                    #i_new = int(i/100)
                    #p = all_states_nocpl[cell_index, i_new]
                    #p = states[cell_index, :]
                    p = c_end[cell_index]
                    #p = dc[cell_index]
                    #print(f"p-value of tuple is", p)
                    #p = cval[cell_index, i_new]
                    vals[idx, idy] = p
                else:
                    #i_new = int(i/100)
                    #p = all_states[cell_index, i_new]
                    #p = cval[cell_index, i_new]
                    p = c_end[cell_index]
                    # p = states[cell_index, :]
                    #p = dc[cell_index]
                    vals[idx,idy] = p
            else:
                pass


    plt.rc('text', usetex=False)
    plt.rc('font', family='serif', size=25)

    plt.figure(figsize=(15,10))

    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([275, 320, -22, 15], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE, linewidth=1)
    ax.coastlines('50m')
    # cmap = plt.get_cmap('turbo')
    # cmap = plt.get_cmap('summer', 20)
    # cmap = matplotlib.cm.ocean(np.linspace(0,1,20))
    # cmap = matplotlib.cm.viridis(np.linspace(0,1,20)) #Workss but doesnt look that good
    # cmap = matplotlib.cm.gist_earth(np.linspace(0,1,20))
    # cmap = matplotlib.cm.summer(np.linspace(0,1,40))
    # cmap = matplotlib.colors.ListedColormap(cmap[2:15])  #Works for viridis [2:20]
    # cmap = matplotlib.colors.ListedColormap(cmap[9:19])  # For gist_earth
    # cmap = matplotlib.colors.ListedColormap(cmap[2:20])
    # cmap = matplotlib.colors.ListedColormap(["darkgreen", "forestgreen","limegreen", "lawngreen", "greenyellow", "darkkhaki", "palegoldenrod", "khaki", "goldenrod", "gold", "yellow"])

    #cmap = matplotlib.colors.LinearSegmentedColormap.from_list('summer', ['#1B5E20','#D4E157','#FFEE58','#FF8F00'])
    
    #cmap = matplotlib.colors.LinearSegmentedColormap.from_list('summer', ['#1B5E20', '#FFEE58']) #For state values in Background Moran's I coefficient
    
    cmap = plt.get_cmap('magma')
    cmap = matplotlib.cm.magma(np.linspace(0,1,20))
    cmap = matplotlib.colors.ListedColormap(cmap[0:17])  #For c_end values
    #cmap = matplotlib.colors.ListedColormap([cmap[0]])

    plt.pcolor(lon-(lon[-1]-lon[-2]) / 2, lat-(lat[-1]-lat[-2]) / 2, vals, cmap=cmap) #For state values

    #nx.draw_networkx(net,pos, edge_color='black', node_size=0, with_labels=False)
    # cbar = plt.colorbar(label='Critical thresholds')
    #cbar = plt.colorbar(label='Dc values')

    #cbar = plt.colorbar(label='State values of nodes')
    cbar = plt.colorbar(label='Critical function values')
    cbar.ax.tick_params(labelsize=25)
    cbar.set_label(label='$F_{crit_{end}}$', size='large')
    #if region == 1:
    #    cbar.set_ticks(np.arange(0, 4.5))
    #else:
    #    pass

    #cmap.set_over('red')
    #if region == 0:
    #    pass
    #else:
    #    plt.clim(0, 0.004)

    #plt.clim(-1.0, -0.5)

    #plt.savefig("critical_thresholds/dcvalues_region{}_v2.png".format(region), bbox_inches='tight')
    
    #plt.savefig("/p/projects/dominoes/lena/jobs/results/noise/final/statevalues_random_region{}.png".format(region), bbox_inches='tight')  #For state values in Background Moran's I coefficient
    plt.savefig("/p/projects/dominoes/lena/jobs/results/noise/final/c_end_{}_region{}_final.png".format(year, region), bbox_inches='tight')
    print(f"Figure saved")

    plt.clf()
    plt.close() 



'''
#Network is created using the monthly data, the critical mcwd, coupling switch and the rain factor
net = generate_network(data_crit, data_crit_std, data_eval, no_cpl_dummy, rain_fact, adapt_fact)


###MAIN - PREPARATION###
output = []

initial_state = np.zeros(net.number_of_nodes())
initial_state.fill(-1)

ev = evolve( net , initial_state )
tolerance = 0.01
#cells = range(0, net.number_of_nodes())
ev_array = np.empty((len(list(cells))+1,1))
i_lis = []


###To get c-values####
for i in itertools.count():
    c_list = []
    #if net.get_tip_states(ev.get_timeseries()[1][-1])[:].any() == False:
    print(f"Net_get timeseries looks like", net.get_tip_states(ev.get_timeseries()[1][-1])[:])
    
    for x in range(0, len(list(cells))):
        #if net.nodes[x]['data'].get_par()['c'] <= c_end[x]:
        c = net.nodes[cells[x]]['data'].get_par()['c']
        # print(f"C-Values are,", c)
        net.set_param(cells[x], 'c', c+dc[x])
        c_list.append(c)
    i_lis.append(i)

    if all(c_list) <= c_end.all(): 
    # Append c to list so can be plotted on x-axis against spatial variance

        ev = evolve (net, ev.get_timeseries()[1][-1])   # making new state as dynamical system for every increased c value
        
        #After evolve the sigmas come in

        sigmas = np.array([0.01] * np.ones(len(list(cells))))
        alphas = 2 * np.ones(len(list(cells)))
        
        ev.integrate(t_step, realtime_break, sigmas = sigmas, noise = "normal", alphas = alphas, seed = None)

        c_arr = np.array(1)
        for j in range(0, len(list(cells))): 
            c_all = net.nodes[cells[j]]['data'].get_par()['c']
            print(f"C_all looks like", c_all)
            c_arr = np.append(c_arr, c_all)
        
        c_arr = c_arr[:, np.newaxis]
        ev_array = np.append(ev_array, c_arr, axis=1)
        print(f"Ev Aarray looks like:", ev_array)
        print("Ilist for year {} and region {}".format(year, region), len(i_lis))
    else:
        break


#else:
#    break


ev_array = ev_array[1:, 1:]
print(f"shape of ev_array is", np.shape(ev_array))
#Save values for individual years
path = '/p/projects/dominoes/lena/jobs/results/noise/final/{}/critical_thresholds'.format(year, region)
fmt = '%1.9f'
if no_cpl_dummy == True:
    np.savetxt(os.path.join(path, "no_coupling/region{}/cvalues_{}_region{}.txt".format(region, year, region)), ev_array, fmt = fmt)
else:
    np.savetxt(os.path.join(path, "region{}/cvalues_{}_region{}.txt".format(region, year, region)), ev_array, fmt = fmt)
print(f"Finished saving c_values")


###To GET PLOTS FOR DIFFERENT NOISE VALUES###
noise = [0.01, 0.02, 0.04, 0.06, 0.08, 0.1]

for item in noise: 

    if no_cpl_dummy == True:
        os.chdir("/p/projects/dominoes/lena/jobs/results/noise/final/{}/noise_states/no_coupling/region{}".format(year, region))
        states = np.loadtxt("states_cusps_{}_region{}_noise{}.txt".format(year, region, item), dtype = np.float64)
    else:
        os.chdir("/p/projects/dominoes/lena/jobs/results/noise/final/{}/noise_states/region{}".format(year, region))
        states = np.loadtxt("states_cusps_{}_region{}_noise{}.txt".format(year, region, item), dtype = np.float64)
    #for i in np.arange(0, ilist[-1]+1, 100):
    num_cols = states.shape[1]
    cols_list = range(0, num_cols)
    print(f"Last element of column list is", states[:, num_cols-1])
    print(f"Cols list", cols_list)
    print(f"num_cols is", num_cols)
    print(f"The middle elements of cols list is:", int(len(cols_list)/2))
    for i in cols_list:
        #if i == ilist[20]: #i == ilist[-2] or i == ilist[-1]:  #i % 100 == 0 or  #For 2005 i % 500, for 2007/2010 i %100
        if (i == cols_list[int(len(cols_list)/2)] or i == cols_list[-2] or i == cols_list[-1]):
            
            #print(f"Vorletzter Value of ilist is", ilist[-2])
            #print(f"Letzter Value of ilist is, ", ilist[-1])
            print(f"The last entry is", i)
            #os.chdir("/p/projects/dominoes/lena/jobs/results/noise/final/{}/critical_thresholds".format(year))
            print("Plotting sequence")
            #latlon values
            lat = net_data.variables["lat"][:]
            lon = net_data.variables["lon"][:]   
            tuples = [(lat[idx],lon[idx]) for idx in range(lat.size)]
            
            
            if no_cpl_dummy == True:   
                cval = np.loadtxt("no_coupling/region{}/cvalues_{}_region{}.txt".format(region, year, region), dtype = np.float64)
            else:
                cval = np.loadtxt("region{}/cvalues_{}_region{}.txt".format(region, year, region), dtype = np.float64)
            

            #print(f"Shape of cval is", np.shape(cval))
    

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
                        #print(f"Tuples index is.,", tuples.index((x,y)))
                        cell_index = list(cells).index(tuples.index((x,y)))
                        #print(f"Cell index is:,", cell_index)
                        #print(f"Last column of cval is", cval[:, 93])
                        
                        #c_value = net.nodes[cell_index]['data'].get_par()['c']
                        #print(f"(C-value, cell_index is", (c_value, cell_index))

                        #Get all cells values if tipped or not in varible p
                        #p = net.get_tip_states(ev.get_timeseries()[1][-1])[:][tuples.index((x,y))]
                        
                        #Get also intermediate values for cells
                        if no_cpl_dummy == True:
                            #i_new = int(i/100)
                            #p = all_states_nocpl[cell_index, i_new]
                            p = states[cell_index, i]
                            #print(f"p-value of tuple is", p)
                            #p = cval[cell_index, i_new]
                            vals[idx, idy] = p
                        else:
                            #i_new = int(i/100)
                            #p = all_states[cell_index, i_new]
                            #p = cval[cell_index, i_new]
                            p = states[cell_index, i]
                            vals[idx,idy] = p
                    else:
                        pass


            plt.rc('text', usetex=False)
            plt.rc('font', family='serif', size=25)

            plt.figure(figsize=(15,10))

            ax = plt.axes(projection=ccrs.PlateCarree())
            ax.set_extent([275, 320, -22, 15], crs=ccrs.PlateCarree())
            ax.add_feature(cfeature.COASTLINE, linewidth=1)
            ax.coastlines('50m')
            # cmap = plt.get_cmap('turbo')
            # cmap = plt.get_cmap('summer', 20)
            # cmap = matplotlib.cm.ocean(np.linspace(0,1,20))
            # cmap = matplotlib.cm.viridis(np.linspace(0,1,20)) #Workss but doesnt look that good
            # cmap = matplotlib.cm.gist_earth(np.linspace(0,1,20))
            # cmap = matplotlib.cm.summer(np.linspace(0,1,40))
            # cmap = matplotlib.colors.ListedColormap(cmap[2:15])  #Works for viridis [2:20]
            # cmap = matplotlib.colors.ListedColormap(cmap[9:19])  # For gist_earth
            # cmap = matplotlib.colors.ListedColormap(cmap[2:20])
            # cmap = matplotlib.colors.ListedColormap(["darkgreen", "forestgreen","limegreen", "lawngreen", "greenyellow", "darkkhaki", "palegoldenrod", "khaki", "goldenrod", "gold", "yellow"])
            cmap = matplotlib.colors.LinearSegmentedColormap.from_list('summer', ['#1B5E20','#D4E157','#FFEE58','#FF8F00'])

            plt.pcolor(lon-(lon[-1]-lon[-2]) / 2, lat-(lat[-1]-lat[-2]) / 2, vals, cmap=cmap)
            #nx.draw_networkx(net,pos, edge_color='black', node_size=0, with_labels=False)
            
            #cbar = plt.colorbar(label='Critical thresholds')
            cbar = plt.colorbar(label='State values')
            cmap.set_over('red')
            plt.clim(-1.0, -0.50)
            #plt.clim(0, np.sqrt(4/27))

            
            if no_cpl_dummy == True:
                if i == cols_list[-2]:
                    plt.savefig("beforetip_noise{}_unstableamazon{:04n}".format(int(item*100), i), bbox_inches='tight')
                elif i == cols_list[-1]:
                    plt.savefig("aftertip_noise{}_unstableamazon{:04n}".format(int(item*100), i), bbox_inches='tight')
                else:
                    plt.savefig("middle_noise{}_unstableamazon{:04n}".format(int(item*100), i), bbox_inches='tight')
            else:
                if i == cols_list[-2]:
                    plt.savefig("beforetip_noise{}_unstableamazon{:04n}".format(int(item*100), i), bbox_inches='tight')
                elif i == cols_list[-1]:
                    plt.savefig("aftertip_noise{}_unstableamazon{:04n}".format(int(item*100), i), bbox_inches='tight')
                else:
                    plt.savefig("middle_noise{}_unstableamazon{:04n}".format(int(item*100), i), bbox_inches='tight')            
            #plt.show()
            
            #When plotting for different noise states
            #plt.savefig("beforetip_noise{}_unstableamazon{:04n}".format(int(item*100), i), bbox_inches='tight')
            
            plt.clf()
            plt.close() 


###TO GET IMAGES FOR DIFFERENT COUPLING VALUES####
coupling = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]

for item in coupling: 

    if no_cpl_dummy == True:
        os.chdir("/p/projects/dominoes/lena/jobs/results/noise/final/{}/coupling_states/no_coupling/region{}".format(year, region))
        states = np.loadtxt("states_cusps_{}_region{}.txt".format(year, region), dtype = np.float64)
    else:
        os.chdir("/p/projects/dominoes/lena/jobs/results/noise/final/{}/coupling_states/region{}".format(year, region))
        states = np.loadtxt("states_cusps_{}_region{}_coupling{}.txt".format(year, region, item), dtype = np.float64)
    #for i in np.arange(0, ilist[-1]+1, 100):
    num_cols = states.shape[1]
    cols_list = range(0, num_cols)
    print(f"Last element of column list is", states[:, num_cols-1])
    print(f"Cols list", cols_list)
    print(f"num_cols is", num_cols)
    print(int(len(cols_list)/2))
    for i in cols_list:
        if (i == cols_list[int(len(cols_list)/2)] or i == cols_list[-2] or i == cols_list[-1]): #i == ilist[-2] or i == ilist[-1]:  #i % 100 == 0 or  #For 2005 i % 500, for 2007/2010 i %100
        #if i == cols_list[-1]:
            
            #print(f"Vorletzter Value of ilist is", ilist[-2])
            #print(f"Letzter Value of ilist is, ", ilist[-1])
            print(f"The last entry is", i)
            #os.chdir("/p/projects/dominoes/lena/jobs/results/noise/final/{}/critical_thresholds".format(year))
            print("Plotting sequence")
            #latlon values
            lat = net_data.variables["lat"][:]
            lon = net_data.variables["lon"][:]   
            tuples = [(lat[idx],lon[idx]) for idx in range(lat.size)]
            
            
            if no_cpl_dummy == True:   
                cval = np.loadtxt("no_coupling/region{}/cvalues_{}_region{}.txt".format(region, year, region), dtype = np.float64)
            else:
                cval = np.loadtxt("region{}/cvalues_{}_region{}.txt".format(region, year, region), dtype = np.float64)
            

            #print(f"Shape of cval is", np.shape(cval))
    

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
                        #print(f"Tuples index is.,", tuples.index((x,y)))
                        cell_index = list(cells).index(tuples.index((x,y)))
                        #print(f"Cell index is:,", cell_index)
                        #print(f"Last column of cval is", cval[:, 93])
                        
                        #c_value = net.nodes[cell_index]['data'].get_par()['c']
                        #print(f"(C-value, cell_index is", (c_value, cell_index))

                        #Get all cells values if tipped or not in varible p
                        #p = net.get_tip_states(ev.get_timeseries()[1][-1])[:][tuples.index((x,y))]
                        
                        #Get also intermediate values for cells
                        if no_cpl_dummy == True:
                            #i_new = int(i/100)
                            #p = all_states_nocpl[cell_index, i_new]
                            p = states[cell_index, i]
                            #print(f"p-value of tuple is", p)
                            #p = cval[cell_index, i_new]
                            vals[idx, idy] = p
                        else:
                            #i_new = int(i/100)
                            #p = all_states[cell_index, i_new]
                            #p = cval[cell_index, i_new]
                            p = states[cell_index, i]
                            vals[idx,idy] = p
                    else:
                        pass


            plt.rc('text', usetex=False)
            plt.rc('font', family='serif', size=25)

            plt.figure(figsize=(15,10))

            ax = plt.axes(projection=ccrs.PlateCarree())
            ax.set_extent([275, 320, -22, 15], crs=ccrs.PlateCarree())
            ax.add_feature(cfeature.COASTLINE, linewidth=1)
            ax.coastlines('50m')
            # cmap = plt.get_cmap('turbo')
            # cmap = plt.get_cmap('summer', 20)
            # cmap = matplotlib.cm.ocean(np.linspace(0,1,20))
            # cmap = matplotlib.cm.viridis(np.linspace(0,1,20)) #Workss but doesnt look that good
            # cmap = matplotlib.cm.gist_earth(np.linspace(0,1,20))
            # cmap = matplotlib.cm.summer(np.linspace(0,1,40))
            # cmap = matplotlib.colors.ListedColormap(cmap[2:15])  #Works for viridis [2:20]
            # cmap = matplotlib.colors.ListedColormap(cmap[9:19])  # For gist_earth
            # cmap = matplotlib.colors.ListedColormap(cmap[2:20])
            # cmap = matplotlib.colors.ListedColormap(["darkgreen", "forestgreen","limegreen", "lawngreen", "greenyellow", "darkkhaki", "palegoldenrod", "khaki", "goldenrod", "gold", "yellow"])
            cmap = matplotlib.colors.LinearSegmentedColormap.from_list('summer', ['#1B5E20','#D4E157','#FFEE58','#FF8F00'])

            plt.pcolor(lon-(lon[-1]-lon[-2]) / 2, lat-(lat[-1]-lat[-2]) / 2, vals, cmap=cmap)
            #nx.draw_networkx(net,pos, edge_color='black', node_size=0, with_labels=False)
            # cbar = plt.colorbar(label='Critical thresholds')
            cbar = plt.colorbar(label='State values')
            cmap.set_over('red')
            plt.clim(-1.0, -0.50)
            #plt.clim(0, np.sqrt(4/27))

            
            if no_cpl_dummy == True:
                if i == cols_list[-2]:
                    plt.savefig("beforetip_coupling0001_unstableamazon{:04n}.png".format(i), bbox_inches='tight')
                elif i == cols_list[-1]:
                    plt.savefig("aftertip_coupling0001_unstableamazon{:04n}.png".format(i), bbox_inches='tight')
                else:
                    plt.savefig("middle_coupling0001_unstableamazon{:04n}.png".format(i), bbox_inches='tight')
            else:
                if i == cols_list[-2]:
                    plt.savefig("beforetip_coupling{:04n}_unstableamazon{:04n}.png".format(item, i), bbox_inches='tight')
                elif i == cols_list[-1]:
                    plt.savefig("aftertip_coupling{:04n}_unstableamazon{:04n}.png".format(item, i), bbox_inches='tight')
                else:
                    plt.savefig("middle_coupling{:04n}_unstableamazon{:04n}.png".format(item, i), bbox_inches='tight')            
            #plt.show()
            
            #When plotting for different noise states
            #plt.savefig("beforetip_noise{}_unstableamazon{:04n}".format(int(item*100), i), bbox_inches='tight')
            
            plt.clf()
            plt.close() 
'''



#####No NEED for this part anymore#####
'''   
else:
    for i in np.arange(0, ilist[-1]+1, 100):
        print(i)
        if i == ilist[-2] or i == ilist[-1]: #or i % 1000 == 0:
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
                tuples_region = [(lat,lon)]

            tuple_list = []
            for idx,x in enumerate(lat):
                for idy,y in enumerate(lon):
                    if (x,y) in tuples and (x,y) in tuples_region: 
                        tuple_list.append(tuples.index((x,y)))
                        #print(f"(x,y) are", ((x,y)))
                        print(f"Tuples index is.,", tuples.index((x,y)))
                        cell_index = list(cells).index(tuples.index((x,y)))
                        print(f"Cell index is:,", cell_index)

                        #Get all cells values if tipped or not in varible p
                        #p = net.get_tip_states(ev.get_timeseries()[1][-1])[:][tuples.index((x,y))]
                        
                        #Get also intermediate values for cells
                        if no_cpl_dummy == True:
                            i_new = int(i/100)
                            p = all_states[cell_index, i_new]
                            #print(f"p-value of tuple is", p)
                            vals[idx,idy] = p
                        else:
                            i_new = int(i/100)
                            print(f"i_new is", i_new)
                            p = all_states[cell_index, i_new]
                            #p = c_end[cell_index]
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
            # cmap = plt.get_cmap('summer', 20)
            # cmap = matplotlib.cm.ocean(np.linspace(0,1,20))
            # cmap = matplotlib.cm.summer(np.linspace(0,1,40))
            # cmap = matplotlib.colors.ListedColormap(cmap[0:40])
            #cmap = matplotlib.cm.nipy_spectral(np.linspace(0,1,30))
            #cmap = matplotlib.colors.ListedColormap(cmap[15:25])

            cmap = matplotlib.colors.LinearSegmentedColormap.from_list('summer', ['#1B5E20','#D4E157','#FFEE58','#FF8F00'])

            plt.pcolor(lon-(lon[-1]-lon[-2]) / 2, lat-(lat[-1]-lat[-2]) / 2, vals, cmap=cmap)
            #nx.draw_networkx(net,pos, edge_color='black', node_size=0, with_labels=False)
            cbar = plt.colorbar(label='States of nodes for Amazon rainforest')
            
            #cbar = plt.colorbar(label='Critical parameter c')  #When plotting c-values
            cmap.set_over('red')
            plt.clim(-1.0, -0.60)

            if no_cpl_dummy == True:
                plt.savefig("no_coupling/region{}/unstableamazon{:04n}".format(region, i), bbox_inches='tight')
            else:
                plt.savefig("region{}/unstableamazon{:04n}".format(region, i), bbox_inches='tight')
                
            print(f"figure created")
            #plt.show()
            plt.clf()
            plt.close() 
'''