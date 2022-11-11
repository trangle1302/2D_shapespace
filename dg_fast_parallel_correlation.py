import os
import numpy as np
import pandas as pd
import deepgraph as dg
import multiprocessing
import time


# parameters (change these to control RAM usage)
step_size = 1e4
n_processes = multiprocessing.cpu_count() - 2

# d = "/scratch/users/tle1302/shapemode/covar_sc"
d = "/mnt/c/Users/trang.le/Desktop/shapemode/covar_sc"
X = np.load(f'{d}/samples.npy', mmap_mode='r')
X = (X - X.mean(axis=1, keepdims=True)) / X.std(axis=1, keepdims=True)
X = X[:4000,:]
#X = X.transpose()
n_samples, n_features = X.shape
print(f"Number of samples: {n_samples}, number of features: {n_features}")

# create node table that stores references to the mem-mapped samples
v = pd.DataFrame({'index': range(X.shape[0])})

# connector function to compute pairwise pearson correlations
def corr(index_s, index_t):
    features_s = X[index_s]
    features_t = X[index_t]
    # print(len(index_t), len(index_s), features_s.shape, features_t.shape)
    corr = np.einsum('ij,ij->i', features_s, features_t) / n_samples
    return corr

# index array for parallelization
pos_array = np.array(np.linspace(0, n_samples*(n_samples-1)//2, n_processes), dtype=int)

# parallel computation
def create_ei(i):

    from_pos = pos_array[i]
    to_pos = pos_array[i+1]

    # initiate DeepGraph
    g = dg.DeepGraph(v)

    # create edges
    g.create_edges(connectors=corr, step_size=step_size,
                   from_pos=from_pos, to_pos=to_pos)

    # store edge table
    g.e.to_pickle(f'{d}/correlations/{str(i).zfill(3)}.pickle')
    
# computation
if __name__ == '__main__':
    s = time.time()
    os.makedirs(f"{d}/correlations", exist_ok=True)
    indices = np.arange(0, n_processes - 1)
    p = multiprocessing.Pool()
    for _ in p.imap_unordered(create_ei, indices):
        pass
    
    # store correlation values
    files = os.listdir(f'{d}/correlations/')
    files.sort()
    store = pd.HDFStore(f'{d}/e.h5', mode='w')
    for f in files:
        et = pd.read_pickle(f'{d}/correlations/{f}')
        store.append('e', et, format='t', data_columns=True, index=False)
    store.close()
    print(f"Finished in {(time.time()-s)/3600}h")