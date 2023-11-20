# This script is used to download and reformat the raw data missing from S-BIAD34 EMBL server
# The raw data is downloaded from smb://storage3.ad.scilifelab.se/confocal 
# and saved in individual antibody folders in /data/2Dshapespace/S-BIAD34/Files

import sys
sys.path.append('..')
import os
import pandas as pd
from utils.helpers_os import find_files_with_patterns, file_exists_and_size
import configs.config as cfg
import shutil
import re
import glob

raw_data_dir = "/mnt/storage3/CELL_PROFILING/research/Diana/" # mounted storage3
download_data_dir = "/data/2Dshapespace/S-BIAD34/Files"
# Greiner plate pattern matching
pattern = r"([A-Z]) - (\d+)\(fld (\d+) wv (DAPI|FITC|Cy3|Cy5) - \4\).tif"
channels = {"DAPI": "w1", "FITC": "w2", "Cy3": "w3", "Cy5": "w4"}

xls = pd.ExcelFile(f'{raw_data_dir}/images_IDR.xlsx')
exp_meta = pd.read_excel(xls, 'Sheet4')
df = pd.read_csv(f'{download_data_dir}_redownload/tobe_checked_list.txt',sep=" ",header=None)
df.columns = ['zipfile']
df['antibody'] = [f.split('.')[1][1:] for f in df.zipfile]
exp_meta = exp_meta[exp_meta['Antibody id'].isin(df.antibody)]
# exception Greiner plate 17 and 20
# missing = ['HPA052427', 'HPA039860', 'HPA070499', 'HPA071010', 'HPA006465', 'HPA070217', 'HPA027838', 'HPA052185', 'HPA053470', 'HPA074403', 'HPA057393', 'HPA020289', 'HPA025735', 'HPA054902']
# exp_meta = exp_meta[exp_meta['Antibody id'].isin(missing)]

for i, row in exp_meta.iterrows():
    ab = row['Antibody id']
    # remove all files inside folder, keep the folder
    if os.path.isdir(f"{download_data_dir}/{ab}"):
        cmd = f"rm {download_data_dir}/{ab}/*"
        os.system(cmd)
    else:
        os.mkdir(f"{download_data_dir}/{ab}")
    plate = row.WellPlate.split('_')[1]
    well = row.well
    #print(plate)
    if len(plate) == 8:
        patterns = [well, plate[4:]]
        #print(patterns)
        file_list = find_files_with_patterns(raw_data_dir, patterns)
        print(f"Found {len(file_list)} files")
        for fullpath in file_list:
            if os.path.getsize(fullpath) > (4 * 1024 * 1024):
                basename = os.path.basename(fullpath)
                shutil.copyfile(fullpath, f"{download_data_dir}/{ab}/{basename}")   
                #print(f"Moving to {download_data_dir}/{ab}/{basename}")

    elif len(plate) == 4: # Greiner plate, different naming pattern
        filelist = glob.glob(f"{raw_data_dir}/*/*{plate}*/*.tif")
        if len(filelist) == 0:
            # exception for plate 6717 and 6720 named only as 17 and 20
            filelist = glob.glob(f"{raw_data_dir}/*/*{plate[2:]}*/*.tif")
        for filename in filelist:
            # Extract the original values from the filename
            match = re.search(pattern, filename)
            if match:
                wellrow, wellcol, stage, channel = match.groups()
                if wellrow + wellcol != well:
                    continue
                new_name = f"{plate}_{wellrow}{wellcol}_s{stage}_{channels[channel]}.tif"
                if os.path.getsize(filename) > (4 * 1024 * 1024):
                    shutil.copyfile(filename, f"{download_data_dir}/{ab}/{new_name}")   
                    #print(f"Moving to {download_data_dir}/{ab}/{new_name}")
                if channel == 'Cy5':
                    shutil.copyfile(filename.replace('.tif','_Rescaled.tif'), f"{download_data_dir}/{ab}/{new_name.replace('.tif','_Rescaled.tif')}")   
    else:
        print("Plate name not recognized: ", plate, well)