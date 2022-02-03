# -*- coding: utf-8 -*-
from . import helpers

import sys 
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
import csv
import json
import pickle
import pandas as pd
from datetime import datetime
from collections import defaultdict
import time

def make_points(n):
    '''Generate set of points (sites) on a 2d plane
    using a normal distribution.  
    
    n - number of points

    returns: a set of points (demand) by writing to a file 'demand.csv'

    '''

    np.random.seed(123)

    # mu, sigma = 0, 1 
    # xs = np.random.normal(mu, sigma, n) 
    # ys = np.random.normal(mu, sigma, n) 
    xs = np.random.rand(n) 
    ys = np.random.rand(n) 
    pop = np.random.randint(100, size=n)

    plt.scatter(xs, ys, s=pop/5+0.15)
    plt.xlim(0,1)
    plt.ylim(0,1)

    # convert to lat/lng for saving
    ss = np.vstack([xs.ravel(),ys.ravel(),pop.ravel()]).T

    # add id
    ss = np.append(ss, np.arange(0, ss.shape[0], dtype='int').reshape(ss.shape[0],1), axis=1)
    #ss = np.insert(ss, 0, np.arange(0, ss.shape[0]).reshape(ss.shape[0],1), axis=1)
    #ss[:,:-1] = np.arange(0, ss.shape[0]).reshape(ss.shape[0],1) 
    #print(ss)

    # re-arrange id 
    #permutation = [2, 1, 0]
    #idx = np.empty_like(permutation)
    #idx[permutation] = np.arange(len(permutation))
    #print(ss[:, idx])
    #ss[:] = ss[:, idx]

    #np.array(['d_'+str(x) for x in np.arange(0,10)]).reshape(5,2)
    #dem_id = np.array(['d_' + str(x) for x in np.arange(0, len(xs))]).reshape(len(xs),1)
    #dem_id = list(itertools.chain(dem_id))
    #dem_id = map(str, dem_id)
    #ss = np.append(ss, dem_id, axis=1)
    
    # save to demand.csv
    fmt = ['%f', '%f', '%i', '%i']
    #fmt = ['%i', '%f', '%f']

    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    np.savetxt('demand.csv', ss, delimiter=',', header='x, y, pop, demand_id', comments='', fmt=fmt)
    print(f"file 'demand.csv' of shape: {ss.shape} created at {now}")


def make_grid(q):
    """generate regular square grid over 2d space
    
    q - number of quadrants (cells) along grid axes, 
        yields q * q potential sites

    returns: P(ID, X, Y)
    writes: array to file 'facility.csv'

    """

    n = q + 2 # the start and endpoint are always on edges

    nx, ny = n, n

    x = np.linspace(0,1,nx)
    y = np.linspace(0,1,ny)
    x = x[1:len(x)-1]
    y = y[1:len(y)-1]

    xx, yy = np.meshgrid(x,y)

    ss = np.vstack([xx.ravel(),yy.ravel()]).T

    # add id
    ss = np.append(ss, np.arange(0,ss.shape[0], dtype='int').reshape(ss.shape[0],1), axis=1)
    
    # plot and save a plot
    plt.scatter(ss[:,0],ss[:,1], marker='x')
    plt.xlim(0,1)
    plt.ylim(0,1)

    # save to file
    fmt = ['%f', '%f', '%i']
    np.savetxt('facility.csv', ss, delimiter=',', header='x,y,facility_id', comments='', fmt=fmt)


def calculate_dist(): 
    """calculate distance via scipy.distance.cdist 

    no params

    saves: distance_matrix.csv to a file in root 
    """

    facility = np.genfromtxt('facility.csv', delimiter=',', skip_header=1)
    demand = np.genfromtxt('demand.csv', delimiter=',', skip_header=1)

    dist = cdist(facility[:,0:2], demand[:,0:2], 'euclidean')

    np.savetxt("distance_matrix.csv", dist, delimiter=",")


def get_covered(r):
    """For each facility find the demand points that are covered given 
    a distance threshold radius (r)
    
    r(int) - radius 

    saves: covered in dict format like so:
    1: [1,2,3]
    2: [1]

    """
    dist_matrix = np.genfromtxt('distance_matrix.csv', delimiter=',')
    demand = np.genfromtxt('demand.csv', delimiter=',', skip_header=1)

    rows = dist_matrix.shape[0]
    cols = dist_matrix.shape[1]

    lst_array = []
    #dict_fac = {}
    dict_fac = defaultdict(list)
    dict_dem = {}

    for i in range(0,rows): # for each potential facility (n=100)
        for j in range(0,cols): # for each demand point (n=50)
            lst_array.append([i,j,dist_matrix[i,j]])
            #print(i, j, dist_matrix[i,j])
            
            # if within threshold add to dictionary 
            if dist_matrix[i,j]<=r: 
                dict_fac[i].append(j)
            # if dist_matrix[i,j]<=r and i in dict_fac: 
            #     #key = i
            #     #dict_fac.setdefault(key, []).append(j)
            #     dict_fac[i].append(j)
            # elif dist_matrix[i,j]<=r:
            #     dict_fac[i] = [j]

    stacked = np.vstack(lst_array)

    # run through the dictionary and calculate total covered pop
    list_pop = []
    for k,v in dict_fac.items():
        s = 0
        for i in v:
            s = s + demand[i,2]
        #dict_dem[k] = s
        #list_pop.append([k, s])
        list_pop.append([k, s, len(v)]) # adds count of covered demand points
    #print(dict_dem)
    stacked_pop = np.vstack(list_pop)

    # convert to array 
    #array_of_total = np.array(list(dict_fac.items()), dtype=dtype)
    #array_of_total = np.fromiter(dict_fac.items(), dtype=float, count=len(dict_fac))
    #print(array_of_total)
    # save to file 
    fmt = ['%i', '%f', '%i']
    np.savetxt('total_pop.csv', stacked_pop, delimiter=',', header='facility,total_pop, cnt_demand', comments='', fmt=fmt)            

    # add id
    #stacked = np.append(stacked, np.arange(0,stacked.shape[0]).reshape(stacked.shape[0],1), axis=1)
    
    # save pairwise distance to file 
    fmt = ['%i', '%i', '%f']
    np.savetxt('pairwise_dist.csv', stacked, delimiter=',', header='facility,demand,dist', comments='', fmt=fmt)

    # save dict to file 
    # with open('covered.txt', 'w') as file:
    #     file.write(json.dumps(dict_fac)) # use `json.loads` to do the reverse
    with open('covered.csv', 'w') as csv_file:  
        writer = csv.writer(csv_file)
        for key, value in dict_fac.items():
            writer.writerow([key, value])
    
    with open('covered.pickle', 'wb') as handle:
        pickle.dump(dict_fac, handle, protocol=pickle.HIGHEST_PROTOCOL)


def naive_greedy(n_sited, sort_par='cnt'):
        """implements a naive greedy search 
        to yield feasible solutions for n-sites maximizing 
        covered demand
        
        n_sited(int): number of facilities to site 
        sort_par: sorting variable 
        
        saves: sited_facilities.csv

        """

        orig_stdout = sys.stdout
        f = open('out.txt', 'w')
        sys.stdout = f

        start_time = time.time()

        # read files with info 
        dtype = [('facility', int), ('population', float), ('cnt', int)]
        facility = np.genfromtxt('facility.csv', delimiter=',', skip_header=1)
        demand = np.genfromtxt('demand.csv', delimiter=',', skip_header=1)
        total_pop = np.genfromtxt('total_pop.csv', delimiter=',', skip_header=1, dtype=dtype)
        with open('covered.pickle', 'rb') as handle:
            coverage = pickle.load(handle)

        # sort total population for looping
        if sort_par == 'cnt':
            sorted_pop = np.sort(total_pop, order=['cnt', 'population']) # add another level of sorting
            sorted_pop = sorted_pop[::-1]
            print(sorted_pop[:5])
        else:
            sorted_pop = np.sort(total_pop, order=['population']) # add another level of sorting
            sorted_pop = sorted_pop[::-1]
            print(sorted_pop[:5]) 
        

        # initiate vars 
        candidate_facilities = total_pop['facility']
        #print(type(candidate_facilities))
        sited_facilities = []
        #all_demand
        covered_demand = []
        temp_covered_demand = []

        # set objective to a zero 
        obj = 0 

        # set the required number of sited facilities 
        p = 0

        # generate initial guess with sorted array 
        # loop over potential facilities 
        for i in sorted_pop:
            #print(i)

            if p>=n_sited:
                break 

            else: 

                # save to temp covered
                temp_covered = coverage[i[0]]
                #print(temp_covered)
                temp_covered_demand.append(temp_covered)
                # flatten the list
                flat_demand =  [item for sublist in temp_covered_demand for item in sublist]
                # convert to set for objective calulation
                uniq_demand = set(flat_demand)
                print('unique: ', uniq_demand)
                #print('length of temp covered', len(uniq_demand))

                # calculate total demand covered
                s = 0
                for u in uniq_demand:
                    s = s + demand[u,2]
                #print(s)

                # compare to obj
                if s>obj:
                    obj = s
                    p = p + 1
                    sited_facilities.append(i[0]) # site facility
                    covered_demand = list(uniq_demand)#temp_covered_demand
                    #print('length of current covered', len(covered_demand))
                    
                    print(f'New solution found with objective value {obj}')
                    print(f'The sited facility_id is {i[0]}')
                else:
                    print('bad solution')
                    # if the solution is inferior, remove covered demand
                    pass

                print('----------------------------------------------')

        print('facilities sited at the following locations:')
        print(sited_facilities)
        print(f'final objective value: {obj}')
        print(f'percentage of population covered: {obj/sum(demand[:,2])}')
        print(f'Completed in {(time.time() - start_time)} seconds')

        # save sited_id 
        #np.savetxt('test.csv', [61, 33, 53, 62, 32, 6, 71, 23, 34, 10, 93], delimiter=',', fmt = '%i')
        np.savetxt('sited_id.csv', sited_facilities, delimiter=',', header='facility', comments='', fmt='%i')

        # save covered_id 
        #fmt = 'i'
        #flat_covered = [item for sublist in covered_demand for item in sublist]
        np.savetxt('covered_id.csv', covered_demand, delimiter=',', header='demand', comments='', fmt='%i')
        #np.savetxt('test2.csv', list({0, 1, 2, 3, 5, 8, 9, 10, 12, 14, 15, 16, 17, 19, 20, 24, 25, 26, 27, 31, 32, 33, 34, 36, 39, 42, 45, 48, 49}), delimiter=',', fmt = '%i')

        sys.stdout = orig_stdout
        f.close()

        print('facilities sited at the following locations:')
        print(sited_facilities)
        print(f'final objective value: {obj}')
        print(f'percentage of population covered: {obj/sum(demand[:,2])}')
        print(f'Completed in {(time.time() - start_time)} seconds')


def plot_solution(patch_radius):
    """plots solution found by heuristic"""

    facility = np.genfromtxt('facility.csv', delimiter=',', skip_header=1)
    demand = np.genfromtxt('demand.csv', delimiter=',', skip_header=1)    
    sited_id = np.genfromtxt('sited_id.csv', delimiter=',', skip_header=1, dtype=int)
    covered_id = np.genfromtxt('covered_id.csv', delimiter=',', skip_header=1, dtype=int)

    #print(facility.shape) 
    print(sited_id.shape)
    print(covered_id.shape)
    #print(sited_id.dtype)

    #print(facility[sited_id,:])

    fig, ax = plt.subplots(figsize=(10,10))

    # plot buffers 
    for x,y in zip(facility[sited_id,0], facility[sited_id,1]):
        circle1 = plt.Circle((x,y), radius = patch_radius, color='b', alpha=.1)
        ax.add_patch(circle1)

    # plot facilities
    #ax.scatter(facility[:,0], facility[:,1], s = facility[:,2], color='k')
    ax.scatter(facility[sited_id,0], facility[sited_id,1], color='b', marker='x')

    # plot demand
    ax.scatter(demand[:,0], demand[:,1], s = demand[:,2], color='k')
    ax.scatter(demand[covered_id,0], demand[covered_id,1], s=demand[covered_id,2], color='red')

    # label demand points
    for txt in covered_id:
        ax.annotate(txt, demand[txt,0:2], color='r', va='bottom', ha='left')

    # label facility points
    for txt in sited_id:
        ax.annotate(txt, facility[txt,0:2], color='b', va='top', ha='right')

    
    #circle1 = plt.Circle(facility[sited_id,0], facility[sited_id,1], 0.1, color='g')

    ax.set_xlim(0,1)
    ax.set_ylim(0,1)
    ax.set_title('Maximal Covering Location Problem', fontsize=14)


