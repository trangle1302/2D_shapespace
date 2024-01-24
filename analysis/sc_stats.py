import skimage
import imageio
import numpy as np
import glob
import time
import sys
sys.path.append("..")
from colocalization_quotient import colocalization_quotient
from scipy.stats import pearsonr
import pandas as pd
import tqdm
import os

def check_size(image, shape, d_type="uint16", max_val=65535):
    if image.shape != shape:
        image = (skimage.transform.resize(image, shape) * max_val).astype(d_type)
    return image

def rescale_intensity(image, min_val=0, max_val=65535):
    dtype = image.dtype
    maxmax = (65535 if dtype == "uint16" else (255 if dtype == "uint8" else 1))
    normed = (image - min_val) / (max_val - min_val)
    return (normed*maxmax).astype(dtype)

def compute_quantiles(image_paths, quantiles=[1, 99]):
    results = []
    for img in image_paths:
        try:
            image = imageio.imread(img)
            results += [np.percentile(image, quantiles)]
        except:
            print(f"Error reading {img}")
    return np.array(results)

def get_sc_statistics_fucci(
    cell_mask, nuclei_mask, mt, gmnn, cdt1, protein, cell_mask_path
):
    regions_c = skimage.measure.regionprops(cell_mask)
    regions_n = skimage.measure.regionprops(nuclei_mask)

    ab_id = cell_mask_path.split("/")[-2]
    img_id = cell_mask_path.split("/")[-1].replace("_cellmask.png", "")
    lines = []
    for region_c in regions_c:
        # Mask of the cell
        cell_area = region_c.area
        minr, minc, maxr, maxc = region_c.bbox
        mask = cell_mask[minr:maxr, minc:maxc].astype(np.uint8)
        mask[mask != region_c.label] = 0
        mask[mask == region_c.label] = 1
        # protein in the whole cell area
        pr = protein[minr:maxr, minc:maxc].copy()
        pr[mask != 1] = 0
        pr_sum = pr.sum()

        mt_ = mt[minr:maxr, minc:maxc].copy()
        mt_[mask != 1] = 0
        mt_sum = mt_.sum()
        mt_mean = mt_sum / cell_area

        # Mask of the nucleus
        cell_label = region_c.label
        region_n = [r for r in regions_n if r.label == cell_label][0]
        nu_area = region_n.area
        minr, minc, maxr, maxc = region_n.bbox
        mask_n = nuclei_mask[minr:maxr, minc:maxc].astype(np.uint8)
        mask_n[mask_n != region_n.label] = 0
        mask_n[mask_n == region_n.label] = 1
        # protein in the nucleus
        pr_nu = protein[minr:maxr, minc:maxc].copy()
        pr_nu[mask_n != 1] = 0
        pr_nu_sum = pr_nu.sum()

        gmnn_ = gmnn[minr:maxr, minc:maxc].copy()
        gmnn_[mask_n != 1] = 0
        gmnn_sum = gmnn_.sum()

        cdt1_ = cdt1[minr:maxr, minc:maxc].copy()
        cdt1_[mask_n != 1] = 0
        cdt1_sum = cdt1_.sum()
        line = (
            ",".join(
                map(
                    str,
                    [
                        ab_id,
                        img_id + "_" + str(cell_label),  # Identifier
                        cell_area,
                        nu_area,
                        region_n.eccentricity,  # Nucleus and cell area
                        pr_sum,
                        pr_nu_sum,  # protein total intensity in whole cell and nucleus region, pr_cytosol_mean = (pr_sum-pr_nu)/(cell_area-nu_area)
                        mt_sum,
                        gmnn_sum,
                        cdt1_sum,
                        region_n.axis_minor_length/region_n.axis_major_length, #aspect_ratio_nu,
                        region_c.axis_minor_length/region_c.axis_major_length, #aspect_ratio_cell
                        colocalization_quotient(protein, mt),
                    ],
                )
            )
            + "\n"
        )
        lines += [line]
    return lines

def get_sc_statistics(cell_mask, nuclei_mask, mt, er, nu, protein, cell_id):
    """ Function to extract single cell statistics from single cell masks
    """
    regions_c = skimage.measure.regionprops(cell_mask)
    regions_n = skimage.measure.regionprops(nuclei_mask)
    assert len(regions_n) == len(regions_n)
    assert len(regions_n) == 1
    region_c = regions_c[0]
    region_n = regions_n[0]

    # Mask of the cell
    cell_area = region_c.area

    # protein in the whole cell area
    protein[cell_mask != 1] = 0
    pr_sum = protein.sum()
    mt[cell_mask != 1] = 0
    mt_sum = mt.sum() 

    # Mask of the nucleus
    nu_area = region_n.area
    # protein in the nucleus 
    nu[nuclei_mask != 1] = 0
    nu_sum = nu.sum()
    protein[nuclei_mask != 1] = 0
    pr_nu_sum = protein.sum()
    er[nuclei_mask != 1] = 0
    er_sum = er.sum()
    
    # protein in nuclear periphery
    #mask_n_dilated = skimage.morphology.dilation(mask_n, skimage.morphology.square(10))
    #mask_periphery = mask_n_dilated - mask_n

    line = (
        ",".join(
            map(
                str,
                [
                    cell_id,  # Identifier
                    cell_area,# Nucleus and cell area
                    nu_area,
                    region_n.eccentricity, # Nucleus eccentricity
                    pr_sum,
                    pr_nu_sum,  # protein total intensity in whole cell and nucleus region, pr_cytosol_mean = (pr_sum-pr_nu)/(cell_area-nu_area)
                    mt_sum, #mt_mean = mt_sum / cell_area
                    er_sum,
                    nu_sum,
                    region_n.axis_minor_length/region_n.axis_major_length, #aspect_ratio_nu,
                    region_c.axis_minor_length/region_c.axis_major_length, #aspect_ratio_cell
                    colocalization_quotient(protein, nu),
                    colocalization_quotient(protein, mt),
                    colocalization_quotient(protein, er),
                    colocalization_quotient(er, mt),
                    colocalization_quotient(nu, mt),
                    colocalization_quotient(nu, er),
                    pearsonr(protein.flatten(), nu.flatten())[0],
                    pearsonr(protein.flatten(), mt.flatten())[0],
                    pearsonr(protein.flatten(), er.flatten())[0],
                    pearsonr(er.flatten(), mt.flatten())[0],
                    pearsonr(nu.flatten(), mt.flatten())[0],
                    pearsonr(nu.flatten(), er.flatten())[0],
                ],
            )
        )
        + "\n"
    )
    return line

def get_sc_statistics_HPA(cell_mask, nuclei_mask, mt, er, nu, protein):
    """ Function to extract single cell statistics from full mask
    """
    remove_size = 100
    clean_small_lines = True
    # Remove nuclei touching the border:
    nuclei_mask = skimage.segmentation.clear_border(nuclei_mask)
    keep_value = np.unique(nuclei_mask)
    borderedcellmask = np.array(
        [[x_ in keep_value for x_ in x] for x in cell_mask]
    ).astype("uint8")
    cell_mask = cell_mask * borderedcellmask
    assert set(np.unique(nuclei_mask)) == set(np.unique(cell_mask))
    lines = []
    for region_c, region_n in zip(
        skimage.measure.regionprops(cell_mask), skimage.measure.regionprops(nuclei_mask)
    ):  
        assert region_c.label == region_n.label
        if region_c.area < remove_size:
            continue
        # draw rectangle around segmented cell and
        # apply a binary mask to the selected region, to eliminate signals from surrounding cell
        minr, minc, maxr, maxc = region_c.bbox

        # get cell mask
        mask = cell_mask[minr:maxr, minc:maxc].astype(np.uint8)
        mask[mask != region_c.label] = 0
        mask[mask == region_c.label] = 1
        if clean_small_lines:  # erose and dilate to remove the small line
            mask = skimage.morphology.erosion(mask, skimage.morphology.square(5))
            mask = skimage.morphology.dilation(mask, skimage.morphology.square(7))
            # get new bbox
            minr_, minc_, maxr_, maxc_ = skimage.measure.regionprops(mask)[0].bbox
            mask = mask[minr_:maxr_, minc_:maxc_]
            minr += minr_
            minc += minc_
            maxr = minr + (maxr_ - minr_)
            maxc = minc + (maxc_ - minc_)

        cell_area = region_c.area
        mt_ = mt[minr:maxr, minc:maxc].copy()
        mt_[mask != 1] = 0
        mt_sum = mt_.sum()
        
        er_ = er[minr:maxr, minc:maxc].copy()
        er_[mask != 1] = 0
        er_sum = er.sum()
        # get protein in the whole cell area
        pr = protein[minr:maxr, minc:maxc].copy()
        pr[mask != 1] = 0
        pr_sum = pr.sum()

        # get nuclei mask
        nu_area = region_n.area
        minr, minc, maxr, maxc = region_n.bbox
        mask_n = nuclei_mask[minr:maxr, minc:maxc].astype(np.uint8)
        mask_n[mask_n != region_n.label] = 0
        mask_n[mask_n == region_n.label] = 1

        # protein in the nucleus         
        pr = protein[minr:maxr, minc:maxc].copy()
        pr[mask_n != 1] = 0
        pr_nu_sum = pr.sum()

        nu_ = nu[minr:maxr, minc:maxc].copy()
        nu_[mask_n != 1] = 0
        nu_sum = pr.sum()

        line = (
            ",".join(
                map(
                    str,
                    [
                        region_n.label,  # Identifier
                        cell_area,# Nucleus and cell area
                        nu_area,
                        region_n.eccentricity, # Nucleus eccentricity
                        pr_sum,
                        pr_nu_sum,  # protein total intensity in whole cell and nucleus region, pr_cytosol_mean = (pr_sum-pr_nu)/(cell_area-nu_area)
                        mt_sum, #mt_mean = mt_sum / cell_area
                        er_sum,
                        nu_sum, # DAPI intensity
                        region_n.axis_minor_length/region_n.axis_major_length, #aspect_ratio_nu,
                        region_c.axis_minor_length/region_c.axis_major_length, #aspect_ratio_cell
                        colocalization_quotient(protein, nu),
                        colocalization_quotient(protein, mt),
                        colocalization_quotient(protein, er),
                        colocalization_quotient(er, mt),
                        colocalization_quotient(nu, mt),
                        colocalization_quotient(nu, er),
                        pearsonr(protein.flatten(), nu.flatten())[0],
                        pearsonr(protein.flatten(), mt.flatten())[0],
                        pearsonr(protein.flatten(), er.flatten())[0],
                        pearsonr(er.flatten(), mt.flatten())[0],
                        pearsonr(nu.flatten(), mt.flatten())[0],
                        pearsonr(nu.flatten(), er.flatten())[0],
                        
                    ],
                )
            )
            + "\n"
        )        
        lines += [line]
    return lines

def main():
    import configs.config as cfg
    d = cfg.PROJECT_DIR
    save_path = f'{cfg.PROJECT_DIR}/single_cell_statistics_rescale_intensity_well99.csv'
    full_FOV_masks = True
    if cfg.CELL_LINE=='S-BIAD34':
        if full_FOV_masks:
            with open(save_path, "a") as f:
            # Save sum quantities and cell+nucleus area, the mean quantities per compartment can be calculated afterwards
                f.write(
                    "ab_id,cell_id,cell_area,nu_area,nu_eccentricity,"+
                    "Protein_cell_sum,Protein_nu_sum,MT_cell_sum,GMNN_nu_sum,CDT1_nu_sum,"+
                    "aspect_ratio_nu,aspect_ratio_cell,coloc_pro_mt\n"
                )
                s = time.time()
                antibodies = os.listdir(f"{d}/cell_masks")
                print(f'Saving to {save_path}')
                for antibody in tqdm.tqdm(antibodies):
                    try: 
                        max_vals = {}
                        for ch in ['w1', 'w2', 'w3', 'w4_Rescaled']:
                            chs = glob.glob(f"{d}/Files/{antibody}/*_{ch}.tif")
                            if len(chs) == 0:
                                continue
                            quantiles = np.array(compute_quantiles(chs, quantiles=[0, 99]))
                            # set xx percentile of a well as max value for all images
                            max_val = quantiles[:, 1].max()
                            max_vals[ch] = max_val
                            #print(f"Max value for {antibody}-{ch} is {max_val}")
                        cell_masks = glob.glob(f"{cfg.PROJECT_DIR}/cell_masks/{antibody}/*_cellmask.png")
                        
                        #print(f"{antibody}: {len(cell_masks)} FOVs found with masks")
                    
                        for cell_mask_path in cell_masks:
                            # Reading all channels and masks
                            cell_mask = imageio.imread(cell_mask_path)
                            nuclei_mask = imageio.imread(
                                cell_mask_path.replace("cellmask", "nucleimask")
                            )
                            ab_id = cell_mask_path.split("/")[-2]
                            img_id = cell_mask_path.split("/")[-1].replace("_cellmask.png", "")
                            mt = imageio.imread(f"{d}/Files/{ab_id}/{img_id}_w1.tif")
                            mt = check_size(mt, cell_mask.shape)
                            mt = rescale_intensity(mt, max_val=max_vals['w1'])
                            gmnn = imageio.imread(f"{d}/Files/{ab_id}/{img_id}_w2.tif")
                            gmnn = check_size(gmnn, cell_mask.shape)
                            gmnn = rescale_intensity(gmnn, max_val=max_vals['w2'])
                            cdt1 = imageio.imread(f"{d}/Files/{ab_id}/{img_id}_w3.tif")
                            cdt1 = check_size(cdt1, cell_mask.shape)
                            cdt1 = rescale_intensity(cdt1, max_val=max_vals['w3'])
                            protein = imageio.imread(f"{d}/Files/{ab_id}/{img_id}_w4_Rescaled.tif")
                            protein = check_size(protein, cell_mask.shape)
                            protein = rescale_intensity(protein, max_val=max_vals['w4_Rescaled'])
                            lines = get_sc_statistics_fucci(
                                cell_mask, nuclei_mask, mt, gmnn, cdt1, protein, cell_mask_path
                            )
                            f.writelines(lines)
                    except Exception as e:
                        print(f"Error processing {antibody}: {e}")
            print(f"Finished in {(time.time()-s)/3600}h")
        else:
            sc_cell_pros = glob.glob(f"{cfg.PROJECT_DIR}/cell_masks/*_protein.png")
            print(sc_cell_pros[:3])
            print(f"Processing {len(sc_cell_pros)} single cells, saving to {save_path}")
            s = time.time()
            with open(save_path, "a") as f:
                # Save sum quantities and cell+nucleus area, the mean quantities per compartment can be calculated afterwards
                f.write(
                    "cell_id,cell_area,nu_area,nu_eccentricity," +
                    "Protein_cell_sum,Protein_nu_sum,MT_cell_sum,GMNN_nu_sum,CDT1_nu_sum,"+
                    "aspect_ratio_nu,aspect_ratio_cell," +
                    "coloc_pro_nu,coloc_pro_mt,coloc_pro_er,coloc_er_mt,coloc_nu_mt,coloc_nu_er," +
                    "pearsonr_pro_nu,pearsonr_pro_mt,pearsonr_pro_er,pearsonr_er_mt,pearsonr_nu_mt,pearsonr_nu_er\n"
                )
                for sc_cell_pro in sc_cell_pros:
                    # Reading all channels and masks
                    cell_shape = np.load(sc_cell_pro.replace("_protein.png", ".npy"))
                    cell_mask = cell_shape[0, :, :]
                    nuclei_mask = cell_shape[1, :, :]
                    protein = imageio.imread(sc_cell_pro)
                    ref = np.load(sc_cell_pro.replace("_protein.png", "_ref.npy")) #mt, er, nu
                    if ref.shape[2] == 3:
                        ref = np.transpose(ref, (2, 0, 1))
                    line = get_sc_statistics(
                        cell_mask, nuclei_mask, ref[0,:,:], ref[1,:,:], ref[2,:,:], protein, os.path.basename(sc_cell_pro).replace("_protein.png","")
                    )
                    f.write(line)
            print(f"Finished in {(time.time()-s)/3600}h")
    else:
        image_dir = "/data/HPA-IF-images"
        mask_dir = "/data/kaggle-dataset/PUBLICHPA/mask/test"
        cell_mask_extension = "cellmask.png"
        nuclei_mask_extension = "nucleimask.png"
        
        ifimages = pd.read_csv(f"{image_dir}/IF-image.csv")
        ifimages = ifimages[ifimages.atlas_name == cfg.CELL_LINE]
        ifimages["ID"] = [f.split("/")[-1][:-1] for f in ifimages.filename]
        im_df = pd.read_csv(f"{mask_dir}.csv")
        imlist = set(im_df.ID.unique()).intersection(set(ifimages.ID))
        print(f"Found {len(imlist)} FOV")
        s = time.time()
        print(f'Saving to {cfg.PROJECT_DIR}/single_cell_statistics.csv')
        with open(save_path, "a") as f:
            # Save sum quantities and cell+nuclseus area, the mean quantities per compartment can be calculated afterwards
            f.write(
                "cell_id,cell_area,nu_area,nu_eccentricity," +
                "Protein_cell_sum,Protein_nu_sum,MT_cell_sum,GMNN_nu_sum,CDT1_nu_sum,"+
                "aspect_ratio_nu,aspect_ratio_cell," +
                "coloc_pro_nu,coloc_pro_mt,coloc_pro_er,coloc_er_mt,coloc_nu_mt,coloc_nu_er," +
                "pearsonr_pro_nu,pearsonr_pro_mt,pearsonr_pro_er,pearsonr_er_mt,pearsonr_nu_mt,pearsonr_nu_er\n"
            )
            for img_id in tqdm.tqdm(imlist, total=len(imlist)):
                cell_mask = imageio.imread(f"{mask_dir}/{img_id}_{cell_mask_extension}")
                nuclei_mask = imageio.imread(f"{mask_dir}/{img_id}_{nuclei_mask_extension}")
                protein = imageio.imread(
                    f"{image_dir}/{img_id.split('_')[0]}/{img_id}_green.png"
                )
                mt = imageio.imread(
                    f"{image_dir}/{img_id.split('_')[0]}/{img_id}_red.png"
                )
                er = imageio.imread(
                    f"{image_dir}/{img_id.split('_')[0]}/{img_id}_yellow.png"
                )
                nu = imageio.imread(
                    f"{image_dir}/{img_id.split('_')[0]}/{img_id}_blue.png"
                )

                lines = get_sc_statistics_HPA(
                    cell_mask, nuclei_mask, mt, er, nu, protein
                )
                f.writelines(lines)
        print(f"Finished in {(time.time()-s)/3600}h")

if __name__ == "__main__":
    main()
