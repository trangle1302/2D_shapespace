import os
import skimage
import imageio
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import glob
import time


def check_size(image, shape, d_type="uint16", max_val=65535):
    if image.shape != shape:
        image = (skimage.transform.resize(image, shape) * max_val).astype(d_type)
    return image


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
                    ],
                )
            )
            + "\n"
        )
        lines += [line]
    return lines


def get_sc_statistics(cell_mask, nuclei_mask, mt, er, protein, cell_mask_path):
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

        er_ = er[minr:maxr, minc:maxc].copy()
        er_[mask_n != 1] = 0
        er_sum = er_.sum()
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
                        er_sum,
                    ],
                )
            )
            + "\n"
        )
        lines += [line]
    return lines


def main():
    d = "/data/2Dshapespace/S-BIAD34"
    cell_masks = glob.glob(f"{d}/cell_masks2/*/*_cellmask.png")
    print(f"{len(cell_masks)} FOVs found with masks")
    s = time.time()
    with open(f"{d}/single_cell_statistics.csv", "a") as f:
        # Save sum quantities and cell+nucleus area, the mean quantities per compartment can be calculated afterwards
        f.write(
            "ab_id,cell_id,cell_area,nu_area,nu_eccentricity,Protein_cell_sum,Protein_nu_sum,MT_cell_sum,GMNN_nu_sum,CDT1_nu_sum\n"
        )
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
            gmnn = imageio.imread(f"{d}/Files/{ab_id}/{img_id}_w2.tif")
            gmnn = check_size(gmnn, cell_mask.shape)
            cdt1 = imageio.imread(f"{d}/Files/{ab_id}/{img_id}_w3.tif")
            cdt1 = check_size(cdt1, cell_mask.shape)
            protein = imageio.imread(f"{d}/Files/{ab_id}/{img_id}_w4_Rescaled.tif")
            protein = check_size(protein, cell_mask.shape)

            lines = get_sc_statistics_fucci(
                cell_mask, nuclei_mask, mt, gmnn, cdt1, protein, cell_mask_path
            )
            f.writelines(lines)
    print(f"Finished in {(time.time()-s)/3600}h")


if __name__ == "__main__":
    main()
