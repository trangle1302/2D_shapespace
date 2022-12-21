import os
import glob
import pandas as pd

imlist = glob.glob("/data/2Dshapespace/S-BIAD34/cell_masks/*/*.npy")
imlist_filtered = [f for f in imlist if os.path.getsize(f)>0]
print(f"Found {len(imlist_filtered)/len(imlist)}% ({len(imlist_filtered)}/{len(imlist)}) segmentations with size >0")

imlist_coefs = []
with open("/data/2Dshapespace/S-BIAD34/fftcoefs/fft_major_axis_polarized_ud/fftcoefs_128.txt") as f:
    for line in f:
        imlist_coefs.append(line.split(',')[0])
overlap = set(imlist_filtered).intersection(set(imlist_coefs))
print(f"Found {len(overlap)}/{len(imlist_coefs)} overlap with available segmentation")

imlist_missing = set(imlist_filtered).difference(set(imlist_coefs))
#pd.DataFrame(imlist_missing).to_csv("/data/2Dshapespace/S-BIAD34/failed_img.csv", index=False)
print(f"{len(imlist_missing)} imags failed fft.")
