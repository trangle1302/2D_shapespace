#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21, 2023
@author: devinsullivan
@author: Anthony J. Cesnik, cesnik@stanford.edu
@author: Trang Le, tle1302@stanford.edu
"""
###################################################################
#### Polar-coordinate pseudo time model & Gaussian Mixture Model
###################################################################
#
import os 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from copy import deepcopy
from scipy.optimize import least_squares
from  matplotlib.colors import LinearSegmentedColormap
from sklearn.mixture import GaussianMixture
from matplotlib.patches import Ellipse

class FucciCellCycle:
    def __init__(self):
        # Length of the cell cycle observed for the FUCCI cell line through live cell imaging exp
        self.G1_LEN = 10.833 #hours (plus 10.833, so 13.458hrs for the S/G2 cutoff)
        self.G1_S_TRANS = 2.625 #hours (plus 10.833 and 2.625 so 25.433 hrs for the G2/M cutoff)
        self.S_G2_LEN = 11.975 #hours (this should be from the G2/M cutoff above to the end)
        self.M_LEN = 0.5 # We excluded M-phase from this analysis
        self.TOT_LEN = self.G1_LEN+self.G1_S_TRANS+self.S_G2_LEN
        self.G1_PROP = self.G1_LEN / self.TOT_LEN
        self.G1_S_PROP = self.G1_S_TRANS / self.TOT_LEN + self.G1_PROP
        self.S_G2_PROP = self.S_G2_LEN / self.TOT_LEN + self.G1_S_PROP

def mvavg(yvals, mv_window):
    '''Calculate the moving average'''
    return np.convolve(yvals, np.ones((mv_window,))/mv_window, mode='valid')
def mvpercentiles(yvals_binned):
    '''Calculate moving percentiles given moving-window binned values'''
    return np.percentile(yvals_binned, [10, 25, 50, 75, 90], axis=1)

def frange(start, stop, step):
    i = start
    while i < stop:
        yield i
        i = round(i+step,14)

def histedges_equalN(x, nbin):
    npt = len(x)
    return np.interp(np.linspace(0, npt, nbin + 1), np.arange(npt), np.sort(x))

def histedges_equalA(x, nbin):
    pow = 0.5
    dx = np.diff(np.sort(x))
    tmp = np.cumsum(dx ** pow)
    tmp = np.pad(tmp, (1, 0), 'constant')
    return np.interp(np.linspace(0, tmp.max(), nbin + 1), tmp, np.sort(x))

def stretch_time(time_data,nbins=1000):
    '''This function is supposed to create uniform density space'''
    n, bins, patches = plt.hist(time_data, histedges_equalN(time_data, nbins), density=True)
    tmp_time_data = deepcopy(time_data)
    trans_time = np.zeros([len(time_data)])
    
    # Get bin indexes
    for i,c_bin in enumerate(bins[1:]):
        c_inds = np.argwhere(tmp_time_data<c_bin)
        trans_time[c_inds] = i/nbins
        tmp_time_data[c_inds] = np.inf
    return trans_time
def calc_R(xc, yc, x, y):
    '''Calculate the distance of each 2D points from the center (xc, yc)'''
    return np.sqrt((x-xc)**2 + (y-yc)**2)
def f_2(c,x,y):
    '''Calculate the algebraic distance between the data points and the mean circle centered at c=(xc, yc)'''
    print(c)
    Ri = calc_R(c[0],c[1],x,y)
    return Ri - Ri.mean()
def cart2pol(x, y):
    '''Convert cartesian coordinates to polar coordinates'''
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)
def pol_sort(inds, more_than_start, less_than_start, *args):
    '''Sort data by polar coordinates and reorder based on the start position of the polar coordinate model'''
    return [np.concatenate((arr[inds][more_than_start], arr[inds][less_than_start])) for arr in args]
def pol2cart(rho, phi):
    '''Apply uniform radius (rho) and convert back'''
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)

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

def calculate_pseudotime(log_gmnn, log_cdt1, save_dir = ""):
    fucci_data = np.column_stack([log_gmnn, log_cdt1])
    x = fucci_data[:,0]
    y = fucci_data[:,1]
    
    print('find center')
    center_estimate = np.mean(x), np.mean(y)
    center_est2 = least_squares(f_2, center_estimate, args=(x, y))
    
    xc_2, yc_2 = center_est2.x
    Ri_2       = calc_R(*center_est2.x,x,y)
    R_2        = Ri_2.mean()
    residu_2   = sum((Ri_2 - R_2)**2)

    # Center data
    centered_data = fucci_data - center_est2.x

    pol_data = cart2pol(centered_data[:,0],centered_data[:,1])
    pol_sort_inds = np.argsort(pol_data[1])
    pol_sort_rho = pol_data[0][pol_sort_inds]
    pol_sort_phi = pol_data[1][pol_sort_inds]
    centered_data_sort0 = centered_data[pol_sort_inds,0]
    centered_data_sort1 = centered_data[pol_sort_inds,1]

    # Rezero to minimum --resoning, cells disappear during mitosis, so we should have the fewest detected cells there
    bins = plt.hist(pol_sort_phi,1000)
    start_phi = bins[1][np.argmin(bins[0])]

    # Move those points to the other side
    more_than_start = np.greater(pol_sort_phi,start_phi)
    less_than_start = np.less_equal(pol_sort_phi,start_phi)
    pol_sort_rho_reorder = np.concatenate((pol_sort_rho[more_than_start],pol_sort_rho[less_than_start]))
    pol_sort_inds_reorder = np.concatenate((pol_sort_inds[more_than_start],pol_sort_inds[less_than_start]))
    pol_sort_phi_reorder = np.concatenate((pol_sort_phi[more_than_start],pol_sort_phi[less_than_start]+np.pi*2))
    pol_sort_centered_data0 = np.concatenate((centered_data_sort0[more_than_start],centered_data_sort0[less_than_start]))
    pol_sort_centered_data1 = np.concatenate((centered_data_sort1[more_than_start],centered_data_sort1[less_than_start]))
    pol_sort_shift = pol_sort_phi_reorder+np.abs(np.min(pol_sort_phi_reorder))

    # Shift and re-scale "time"
    # reverse "time" since the cycle goes counter-clockwise wrt the fucci plot
    pol_sort_norm = pol_sort_shift/np.max(pol_sort_shift)
    pol_sort_norm_rev = 1 - pol_sort_norm 
    pol_sort_norm_rev = stretch_time(pol_sort_norm_rev)
    pol_unsort = np.argsort(pol_sort_inds_reorder)
    fucci_time = pol_sort_norm_rev[pol_unsort]
    if save_dir!="":
        WINDOW_FUCCI_PSEUDOTIME = 100
        fucci = FucciCellCycle()
        plt.figure()
        WINDOW_FUCCI_PSEUDOTIMEs = np.asarray([np.arange(start, start + WINDOW_FUCCI_PSEUDOTIME) for start in np.arange(len(pol_sort_norm_rev) - WINDOW_FUCCI_PSEUDOTIME + 1)])
        mvperc_red = mvpercentiles(pol_sort_centered_data1[WINDOW_FUCCI_PSEUDOTIMEs])
        mvperc_green = mvpercentiles(pol_sort_centered_data0[WINDOW_FUCCI_PSEUDOTIMEs])
        mvavg_xvals = mvavg(pol_sort_norm_rev, WINDOW_FUCCI_PSEUDOTIME)
        plt.fill_between(mvavg_xvals * fucci.TOT_LEN, mvperc_green[1], mvperc_green[-2], color="lightgreen", label="25th & 75th Percentiles")
        plt.fill_between(mvavg_xvals * fucci.TOT_LEN, mvperc_red[1], mvperc_red[-2], color="lightcoral", label="25th & 75th Percentiles")
        
        mvavg_red = mvavg(pol_sort_centered_data1, WINDOW_FUCCI_PSEUDOTIME)
        mvavg_green = mvavg(pol_sort_centered_data0, WINDOW_FUCCI_PSEUDOTIME)
        plt.plot(mvavg_xvals * fucci.TOT_LEN, mvavg_red, color="r", label="Mean Intensity")
        plt.plot(mvavg_xvals * fucci.TOT_LEN, mvavg_green, color="g", label="Mean Intensity")
        plt.xlabel('Cell Cycle Time, hrs')
        plt.ylabel('Log10 Tagged CDT1 & GMNN Intensity')
        plt.xticks(size=14)
        plt.yticks(size=14)
        # plt.ylim(0, 1)
        plt.tight_layout()
        plt.savefig(f"{save_dir}/cellcycle_time.png")
    return fucci_time

def draw_ellipse(position, covariance, ax=None, **kwargs):
    """Draw an ellipse with a given position and covariance"""
    ax = ax or plt.gca()
    
    # Convert covariance to principal axes
    if covariance.shape == (2, 2):
        U, s, Vt = np.linalg.svd(covariance)
        angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
        width, height = 2 * np.sqrt(s)
    else:
        angle = 0
        width, height = 2 * np.sqrt(covariance)
    
    # Draw the Ellipse
    for nsig in range(1, 4):
        ax.add_patch(Ellipse(position, nsig * width, nsig * height,
                             angle, **kwargs))
        
def plot_gmm(gmm, X, label=True, ax=None):
    ax = ax or plt.gca()
    labels = gmm.predict(X)
    if label:
        ax.scatter(X[:, 0], X[:, 1], c=labels, s=2, alpha=0.005, cmap='viridis', zorder=2)
    else:
        ax.scatter(X[:, 0], X[:, 1], s=2, zorder=2, alpha=0.005)
    ax.axis('equal')
    
    w_factor = gmm.weights_.max()
    for pos, covar, w in zip(gmm.means_, gmm.covariances_, gmm.weights_):
        draw_ellipse(pos, covar, alpha= w * w_factor)

def GMM_cellcycle(data):
    gmm = GaussianMixture(n_components=3, covariance_type='full').fit(data)
    labels = gmm.predict(data)
    return gmm, labels 

def main():    
    project_dir = f"/data/2Dshapespace/S-BIAD34"
    #project_dir = "/mnt/c/Users/trang.le/Desktop/shapemode/S-BIAD34"
    sc_stats = pd.read_csv(f"{project_dir}/single_cell_statistics.csv") 
    sc_stats["GMNN_nu_mean"] = sc_stats.GMNN_nu_sum/sc_stats.nu_area
    sc_stats["CDT1_nu_mean"] = sc_stats.CDT1_nu_sum/sc_stats.nu_area
    gmnn = np.log10(sc_stats.GMNN_nu_mean)
    cdt1 = np.log10(sc_stats.CDT1_nu_mean)    

    # >>>>> Gaussian Mixture Model
    pseudotime = calculate_pseudotime(gmnn.copy(), cdt1.copy(), save_dir=project_dir)
    sc_stats["GMNN_nu_mean"] = gmnn
    sc_stats["CDT1_nu_mean"] = cdt1
    sc_stats["pseudotime"] = pseudotime
    colors = LinearSegmentedColormap.from_list('rg',["r", "y", "g"])(pseudotime)
    plt.figure()
    plt.scatter(gmnn, cdt1, s=0.1, c = colors)
    plt.xlabel('log10[GMNN_mean]')
    plt.ylabel('log10[CDT1_mean]')
    plt.savefig(f"{project_dir}/fucci_polar_pseudotime.png")

    # >>>>> Gaussian Mixture Model
    data = np.concatenate(gmnn,cdt1)
    gmm, gmm_labels = GMM_cellcycle(data)
    sc_stats["GMM_cc"] = gmm_labels
    # save output
    sc_stats.to_csv(f"{project_dir}/single_cell_statistics.csv", index=False)

    # Plotting for visualization of cluster assignments
    fig, ax = plt.subplots()
    cdict = {0: 'red', 1: 'green', 2: 'yellow'}
    for g in np.unique(gmm_labels):
        idx = np.where(gmm_labels == g)
        ax.scatter(data[idx, 0], data[idx, 1], c = cdict[g], label = g, s = 0.1, alpha=0.05)
    ax.legend()
    plt.xlabel('log(GMNN)')
    plt.ylabel('log(CDT1)')
    plt.savefig(f"{project_dir}/fucci_GMM_points.png")
    
    plt.figure()
    plot_gmm(gmm, data, label=True, ax=None)
    plt.savefig(f"{project_dir}/fucci_GMM_probs.png")

if __name__ == '__main__':
    main()