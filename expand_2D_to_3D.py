import os
import glob
import numpy as np

d_2d = "../U2OS"
d_3d = os.path.join(d_2d, 'U2OS_3D')
if not os.path.exists(d_3d):
    os.makedirs(d_3d)
imlist = glob.glob(d_2d + "/*.npy")

for im in imlist:
    name = os.path.basename(im)
    data = np.load(im)
    data = np.expand_dims(data, axis=1)
    data = np.repeat(data, 10, axis=1)
    data = np.swapaxes(data, 2,3) #channel,z,h,w
    empty = np.zeros((2,3,data.shape[2],data.shape[3]))
    data = np.hstack((empty, data, empty))
    np.save(f'{d_3d}/{name}.npy', data)