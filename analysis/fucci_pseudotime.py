#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21, 2023
@author: trangle
"""
########################################################
#### Polar-coordinate pseudo time model 
########################################################
# 
import os 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from copy import deepcopy
from scipy.optimize import least_squares

def frange(start, stop, step):
    i = start
    while i < stop:
        yield i
        i = round(i+step,14)

def histedges_equalN(x, nbin):
    npt = len(x)
    return np.interp(np.linspace(0, npt, nbin + 1),
                     np.arange(npt),
                     np.sort(x))

def histedges_equalA(x, nbin):
    pow = 0.5
    dx = np.diff(np.sort(x))
    tmp = np.cumsum(dx ** pow)
    tmp = np.pad(tmp, (1, 0), 'constant')
    return np.interp(np.linspace(0, tmp.max(), nbin + 1),
                     tmp,
                     np.sort(x))

def stretch_time(time_data,nbins=1000):
    #This function is supposed to create uniform density space
    _, bins, patches = plt.hist(time_data, histedges_equalN(time_data, nbins), normed=True)
    #data_hist = plt.hist(time_data,nbins)

    tmp_time_data = deepcopy(time_data)
    # ndecimals = np.ceil(np.log10(nbins))
    # rnd_time_data = np.round(time_data,decimals=int(ndecimals))

    trans_time = np.zeros([len(time_data)])
    for i,c_bin in enumerate(bins[1:]):
        #get curr bin indexs
        c_inds = np.argwhere(tmp_time_data < c_bin)
        trans_time[c_inds] = i/nbins
        tmp_time_data[c_inds] = np.inf

    return trans_time

def f_2(c,x,y):
    """ calculate the algebraic distance between the data points and 
    the mean circle centered at c=(xc, yc) 
    """
    Ri = np.sqrt((x-c[0])**2 + (y-c[1])**2)
    return Ri - Ri.mean()

def cart2pol(x, y):
    """Cartersian to Polar"""
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return (rho, phi)

def pol2cart(rho, phi):
    """Polar to Cartersian"""
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return (x, y)

def center_data(log_green_fucci, log_red_fucci):
    fucci_data = np.column_stack([log_green_fucci,log_red_fucci])
    
    x = fucci_data[:,0] 
    y = fucci_data[:,1]

    print('find center')
    center_estimate = np.mean(fucci_data[:,0]), np.mean(fucci_data[:,1])
    center_2 = least_squares(f_2, center_estimate, args=(x, y))
    
    #Center data
    centered_data = fucci_data-center_2.x
    return centered_data

def calculate_pseudotime(log_gmnn, log_cdt1):
    fucci_data = np.column_stack([log_gmnn, log_cdt1])
    x0 = np.ones(5)
    x = fucci_data[:,0]
    y = fucci_data[:,1]

    t_test = np.linspace(np.min(x),np.max(x))
    
    print('find center')
    center_estimate = np.mean(fucci_data[:,0]), np.mean(fucci_data[:,1])
    center_2 = least_squares(f_2, center_estimate, args=(x, y))
    
    #Calculate average radius
    xc_2, yc_2 = center_2.x
    Ri_2       = np.sqrt((x-xc_2)**2 + (y-yc_2)**2)
    R_2        = Ri_2.mean()

    #Center data
    centered_data = fucci_data-center_2.x
    
    #Convert data to polar
    pol_data = cart2pol(centered_data[:,0],centered_data[:,1])
    pol_sort_inds = np.argsort(pol_data[1])
    pol_sort_rho = pol_data[0][pol_sort_inds]
    pol_sort_phi = pol_data[1][pol_sort_inds]
    centered_data_sort0 = centered_data[pol_sort_inds,0] #cartesian x
    centered_data_sort1 = centered_data[pol_sort_inds,1] #cartesian y
    
    # plot polar coordinates
    phi = pol_data[1]
    r = pol_data[0]
    area = 200 * r**2
    fig = plt.figure()
    ax = fig.add_subplot(111, polar=True)
    c = ax.scatter(phi, r, c=phi, s=area, cmap='hsv', alpha=0.75)
    
    
    #rezero to minimum --resoning, cells disappear during mitosis, so we should have the fewest detected cells there
    bins = plt.hist(pol_sort_phi,1000)
    start_phi = bins[1][np.argmin(bins[0])] #pseudotime = 0 and 1

    #move those points to the other side
    more_than_start = np.greater(pol_sort_phi,start_phi)
    less_than_start = np.less_equal(pol_sort_phi,start_phi)
    pol_sort_rho_reorder = np.concatenate((pol_sort_rho[more_than_start],pol_sort_rho[less_than_start]))
    pol_sort_inds_reorder = np.concatenate((pol_sort_inds[more_than_start],pol_sort_inds[less_than_start]))
    pol_sort_phi_reorder = np.concatenate((pol_sort_phi[more_than_start],pol_sort_phi[less_than_start]+np.pi*2))
    #pol_sort_centered_data0 = np.concatenate((centered_data_sort0[more_than_start],centered_data_sort0[less_than_start]))
    #pol_sort_centered_data1 = np.concatenate((centered_data_sort1[more_than_start],centered_data_sort1[less_than_start]))
    #pol_sort_fred = np.concatenate((fred_sort[more_than_start],fred_sort[less_than_start]))#+abs(np.min(fred_sort))
    #pol_sort_fgreen = np.concatenate((fgreen_sort[more_than_start],fgreen_sort[less_than_start]))#+abs(np.min(fgreen_sort))

    #shift and re-scale "time"
    pol_sort_shift = pol_sort_phi_reorder+np.abs(np.min(pol_sort_phi_reorder))
    pol_sort_norm = pol_sort_shift/np.max(pol_sort_shift)
    #pol_sort_shift = (pol_sort_shift - min)/(max-min)
    #reverse "time" since the cycle goes counter-clockwise wrt the fucci plot
    pol_sort_norm_rev = 1-pol_sort_norm
    #stretch time so that each data point is 1
    pol_sort_norm_stretch = stretch_time(pol_sort_norm_rev)
    
    #cart_data_ur = pol2cart(np.repeat(R_2,len(centered_data)), pol_data[1])

    if True:
        fig, ax = plt.subplots(1,2, figsize=(20,10))
        ax[0].plot(pol_sort_norm_rev)
        ax[0].set(xlabel='Number of cells', ylabel='Pseudotime')
        ax[1].plot(pol_sort_norm_stretch)
        ax[1].set(xlabel='Number of cells', ylabel='Pseudotime')
    return pol_sort_norm_stretch[pol_sort_inds_reorder]


def main():    
    project_dir = f"/data/2Dshapespace/S-BIAD34"
    sc_stats = pd.read_csv(f"{project_dir}/single_cell_statistics.csv") 
    sc_stats["GMNN_nu_mean"] = sc_stats.GMNN_nu_sum/sc_stats.nu_area
    sc_stats["CDT1_nu_mean"] = sc_stats.CDT1_nu_sum/sc_stats.nu_area
    gmnn = np.log10(sc_stats.GMNN_nu_mean)
    cdt1 = np.log10(sc_stats.CDT1_nu_mean)


    fucci_data = np.column_stack([gmnn, cdt1])
    plt.hist2d(gmnn, cdt1, bins=200)
    
    pseudotime = calculate_pseudotime(gmnn, cdt1)

if __name__ == '__main__':
    main()