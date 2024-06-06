
from utils.helpers import get_line
from scipy.ndimage import center_of_mass, rotate
from scipy.stats import ttest_ind, kruskal, false_discovery_control 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import cv2
import configs.config as cfg

cyto_region = ["Actin filaments", 
               "Aggresome", 
               "Centriolar satellite",
               "Centrosome",
               "Cleavage furrow",
               "Cytokinetic bridge", 
               "Cytoplasmic bodies", 
               "Cytosol", 
               "Focal adhesion sites", 
               "Intermediate filaments", 
               "Microtubule ends", 
               "Microtubules", 
               "Mitochondria", 
               "Midbody",
               "Midbody ring",
               "Mitotic spindle", 
               "Rods & Rings"]
cell_region = ["Peroxisomes", 
               "Cell Junctions",
               "Endoplasmic reticulum",
                "Golgi apparatus",
                "Lipid droplets",
                "Lysosomes",
                "Endosomes",
                "Microtubule organizing center",
                "Plasma membrane",
                "Vesicles"]
nu_region = ["Kinetochore", 
             "Mitotic chromosome", 
             "Nucleoli fibrillar center", 
             "Nucleoli", 
             "Nuclear bodies", 
             "Nuclear membrane", 
             "Nuclear speckles", 
             "Nucleoplasm", 
             "Nucleoli rim"]
 
def get_regions_to_consider(ab_locations):
    try:
        region_to_consider = []
        for ab_location in ab_locations:
            if ab_location in cyto_region:
                region_to_consider += ['cyto']
            elif ab_location in cell_region:
                region_to_consider += ['cell']
            elif ab_location in nu_region:
                region_to_consider += ['nu']
        region_to_consider = set(region_to_consider)
    except:
        region_to_consider = ["cell"]
    feature_names = []
    for region in region_to_consider:
        if region == 'nu':
            feature_names.append('Protein_nu_mean')
        elif region == 'cyto':
            feature_names.append('Protein_cyt_mean')
        elif region == 'cell':
            feature_names.append('Protein_cell_mean')
    return feature_names

def t_tests_on_groups(df, value='Protein_nu_mean'):
    results = []
    for name, group in df.groupby('antibody'):
        feature_names = get_regions_to_consider(str(group.locations.unique()[0]).split(','))
        for value in feature_names:
            groups = [group[group['groups'] == i][value] for i in [0,1,2]]
            t12, p12 = ttest_ind(groups[0], groups[1], equal_var=False)
            t23, p23 = ttest_ind(groups[1], groups[2], equal_var=False)
            t13, p13 = ttest_ind(groups[0], groups[2], equal_var=False)
            n1, n2, n3 = len(groups[0]), len(groups[1]), len(groups[2])
            results.append({
                'antibody': name,
                'gene_names': group.gene_names.unique()[0],
                'location': group.locations.unique()[0],
                'region': value,
                'g1_means': groups[0].mean(), 'g2_means': groups[1].mean(), 'g3_means': groups[2].mean(),
                't_1vs2': t12, 'p_1vs2': p12,
                't_2vs3': t23, 'p_2vs3': p23, 
                't_1vs3': t13, 'p_1vs3': p13,
                'n_group1': n1, 'n_group2': n2, 'n_group3': n3,
            })     
    return pd.DataFrame(results)


def kruskal_wallis_test(df, value='Protein_nu_mean'):
    results = []
    for name, group in df.groupby('antibody'):
        feature_names = get_regions_to_consider(str(group.locations.unique()[0]).split(','))
        for value in feature_names:
            groups = [group[group['groups'] == i][value] for i in [0,1,2]]
            h, p = kruskal(groups[0], groups[1], groups[2])
            n1, n2, n3 = len(groups[0]), len(groups[1]), len(groups[2])
            results.append({
                'antibody': name,
                'gene_names': group.gene_names.unique()[0],
                'location': group.locations.unique()[0],
                'region': value,
                'g1_means': groups[0].mean(), 'g2_means': groups[1].mean(), 'g3_means': groups[2].mean(),
                'H': h, 'p': p,
                'n_group1': n1, 'n_group2': n2, 'n_group3': n3,
            })     
    return pd.DataFrame(results)


############# PLOTTINGS #############

def boxplots_style1(sc_stats, antibody, value = "Protein_nu_mean", save_dir = ""):
    if antibody is None:
        sub_df = sc_stats
        gene_name = 'allgenes'
    else:
        #sub_df = sc_stats[sc_stats.gene_names==gene_name]
        sub_df = sc_stats[sc_stats.antibody==antibody]
        gene_name = sub_df.gene_names.unique()[0]
    print(sub_df.shape)
    sub_df.loc[:, 'log10_Protein_value'] = np.log10(sub_df.loc[:,value] + 0.000001)
    groups = [sub_df[sub_df['groups'] == i][value] for i in [0,1,2]]

    # ax = sns.boxplot(x ='groups', y='log10_Protein_value', data=sub_df, showfliers = False)
    ax = sns.violinplot(x ='groups', y='log10_Protein_value', data=sub_df, color = "skyblue", alpha=0.5)
    ax = sns.swarmplot(x ='groups', y='log10_Protein_value', data=sub_df, color=".25", alpha=0.5)
    x1, x2, x3 = 0, 1, 2   # (first column: 0, see plt.xticks())

    _, pval = ttest_ind(groups[0], groups[2], equal_var=False)
    y, h, col = sub_df.log10_Protein_value.max()*1.05, 0.01, 'k'
    ax.plot([x1, x1, x3, x3], [y, y+h, y+h, y], lw=1.5, c=col)
    ax.text((x1+x3)*.5, y+h, f"p={pval}", ha='center', va='bottom', color=col)
    
    _, pval = ttest_ind(groups[0], groups[1], equal_var=False)
    y, h, col = sub_df.log10_Protein_value.max()*1.03, 0.01, 'k'
    ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
    ax.text((x1+x2)*.5, y+h, f"p={pval}", ha='center', va='bottom', color=col)

    _, pval = ttest_ind(groups[1], groups[2], equal_var=False)
    y, h, col = sub_df.log10_Protein_value.max()*1.01, 0.01, 'k'
    ax.plot([x2, x2, x3, x3], [y, y+h, y+h, y], lw=1.5, c=col)
    ax.text((x2+x3)*.5, y+h, f"p={pval}", ha='center', va='bottom', color=col)
    plt.ylabel(f"log10({value})")
    plt.xticks([0,1,2], [f'b1 \n n={len(groups[0])}', f'b2 \n n={len(groups[1])}', f'b3 \n n={len(groups[2])}'])
    plt.savefig(f"{save_dir}/{gene_name}_{antibody}_boxplot_{value.split('_')[1]}.png", dpi=300)
    plt.close()

def boxplots_style2(sc_stats, antibody, value = "Protein_nu_mean", save_dir = ""):
    if antibody is None:
        sub_df = sc_stats
        gene_name = 'allgenes'
    else:
        sub_df = sc_stats[sc_stats.antibody==antibody]
        gene_name = sub_df.gene_names.unique()[0]
    print(sub_df.shape)
    sub_df.loc[:, 'log10_Protein_value'] = np.log10(sub_df.loc[:,value] + 0.000001)
    groups = [sub_df[sub_df['groups'] == i][value] for i in [0,1,2]]
    kw_result = kruskal(groups[0], groups[1], groups[2])

    plt.title(f'Kruskal-Wallis Test: H = {kw_result.statistic:.2f}, p = {kw_result.pvalue:.2e}')

    ax = sns.boxplot(x ='groups', y='log10_Protein_value', data=sub_df, fill=False, color = "skyblue", showfliers = False)
    #ax = sns.violinplot(x ='groups', y='log10_Protein_value', data=sub_df, color = "skyblue", alpha=0.5)
    ax = sns.swarmplot(x ='groups', y='log10_Protein_value', data=sub_df, color=".25", alpha=0.5)
    x1, x2, x3 = 0, 1, 2   # (first column: 0, see plt.xticks())

    _, pval02 = ttest_ind(groups[0], groups[2], equal_var=False)
    _, pval01 = ttest_ind(groups[0], groups[1], equal_var=False)
    _, pval12 = ttest_ind(groups[1], groups[2], equal_var=False)
    padjs = false_discovery_control([pval02, pval01, pval12], method='BH')

    y, h, col = sub_df.log10_Protein_value.max()*1.05, 0.01, 'k'
    ax.plot([x1, x1, x3, x3], [y, y+h, y+h, y], lw=1.5, c=col)
    ax.text((x1+x3)*.5, y+h, f"padj={padjs[0]:.1E}", ha='center', va='bottom', color=col)
    
    y, h, col = sub_df.log10_Protein_value.max()*1.03, 0.01, 'k'
    ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
    ax.text((x1+x2)*.5, y+h, f"padj={padjs[1]:.1E}", ha='center', va='bottom', color=col)

    y, h, col = sub_df.log10_Protein_value.max()*1.01, 0.01, 'k'
    ax.plot([x2, x2, x3, x3], [y, y+h, y+h, y], lw=1.5, c=col)
    ax.text((x2+x3)*.5, y+h, f"padj={padjs[2]:.1E}", ha='center', va='bottom', color=col)
    plt.ylabel(f"log10({value})")
    plt.xticks([0,1,2], [f'b1 \n n={len(groups[0])}', f'b2 \n n={len(groups[1])}', f'b3 \n n={len(groups[2])}'])
    plt.savefig(f"{save_dir}/{gene_name}_{antibody}_boxplot_{value.split('_')[1]}.png", dpi=300)
    plt.close()


def resize_with_padding(img, expected_size):
    # expected size need to be larger or equal to than image
    img = cv2.resize(img, (img.shape[1]//5, img.shape[0]//5), interpolation=cv2.INTER_CUBIC)
    assert (expected_size[0] >= img.shape[0]) and (expected_size[1] >= img.shape[1])
    delta_width = expected_size[1] - img.shape[1]
    delta_height = expected_size[0] - img.shape[0]
    pad_width = delta_width // 2
    pad_height = delta_height // 2

    # Create an empty array of the expected size
    padded_image = np.zeros((expected_size[1], expected_size[0],3), dtype=np.uint8)

    # Calculate the position to place the image in the center
    y1, y2 = pad_height, pad_height + img.shape[0]
    x1, x2 = pad_width, pad_width + img.shape[1]

    # Insert the image into the empty array
    padded_image[y1:y2, x1:x2,:] = img

    return padded_image

def draw_contour_on_image(binary_mask, image):
    # Find contours in the binary mask
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Draw red contours on the image
    image_with_contour = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB) if len(image.shape) == 2 else image.copy()
    cv2.drawContours(image_with_contour, contours, -1, (0, 0, 255), 3)
    return image_with_contour

def plot_example_images(sc_stats, antibody, save_dir = ""):
    sub_df = sc_stats[sc_stats.antibody==antibody]
    gene_name = sub_df.gene_names.unique()[0]
    #sub_df = sc_stats[sc_stats.gene_names==gene_name]
    nrow=5
    img_size = 300
    groups = [sub_df[sub_df['groups'] == i]['image_path'].sample(n=nrow) if len(sub_df[sub_df['groups'] == i])>=nrow else sub_df[sub_df['groups'] == i]['image_path'] for i in [0,1,2] ]
    fft_shift_path = f"{cfg.PROJECT_DIR}/fftcoefs/{cfg.ALIGNMENT}/shift_error_meta_fft128.txt"
    # Create a 3x3 grid of images for each group
    images = []
    for group in groups:
        for i in range(nrow):
            if True: # try:
                path = group.values[i] 
                img = cv2.imread(path.replace(".npy", "_protein.png"), cv2.IMREAD_GRAYSCALE)             
                img = cv2.applyColorMap(img, cv2.COLORMAP_TURBO)
                cellborders = np.load(path)
                pattern = path.split('/')[-1] # For U2OS or other cell lines in HPA, image name is enough for identifier
                pattern = path.split('/')[-2] + "/" + path.split('/')[-1] # For U2OS-Fucci / S-BIAD34, antibody/image name is the identifier
                print(f"Finding {pattern} from {fft_shift_path} ")
                fft_coefs = get_line(fft_shift_path, search_text=pattern, mode="first")
                vals = fft_coefs.strip().split(";") 
                theta = float(vals[1])
                print(vals[1], theta)
                # Rotate all channels     
                cell_ = rotate(cellborders[0,:,:], theta)   
                nu_ = rotate(cellborders[1,:,:], theta)    
                img = rotate(img, theta)
                center_cell = center_of_mass(cell_)
                center_nuclei = center_of_mass(nu_)
                if (
                    center_cell[1] > center_nuclei[1]
                ):  # Move 2 quadrant counter-clockwise
                    cell_ = rotate(cell_, 180)
                    nu_ = rotate(nu_, 180)
                    img = rotate(img, 180)
                cell_ = rotate(cell_, 90)
                nu_ = rotate(nu_, 90)
                img = rotate(img, 90)
                img = draw_contour_on_image(cell_, img)
                img = draw_contour_on_image(nu_, img)
                img = resize_with_padding(img,(img_size,img_size))
            if False: #except:
                img = np.zeros((img_size,img_size, 3), dtype=np.uint8)
            images.append(img) 
    # Create a collage of images for the group
    collage = cv2.hconcat([cv2.vconcat(images[:nrow]), cv2.vconcat(images[nrow:nrow*2]), cv2.vconcat(images[nrow*2:])])
    # Save the collage
    cv2.imwrite(f"{save_dir}/{gene_name}_{antibody}_example_cells.png", collage)


# Plot a UMAP for each bin
def plot_protein_through_umap(features, labels, pc_cell = [], genes = [], save_dir = ''):
    # contrasting color list
    color_list = [
        "#FF0000", "#00FF00", "#0000FF", "#FFFF00", "#00FFFF",
        "#800080", "#FFA500", "#008080", "#FFC0CB", "#4B0082",
        "#008000", "#800000", "#00FFFF", "#808000", "#000080",
        "#FF00FF", "#C0C0C0", "#808080", "#A52A2A", "#FFD700"
    ]
    n_bins = len(pc_cell)
    fig, ax = plt.subplots(1,n_bins, figsize=(40, 10))
    for i, ls in enumerate(pc_cell):
        cell_ids = [os.path.basename(f).split('.')[0] for f in ls]
        #features_tmp = features[labels.cell_idx.isin(cell_ids),:]
        labels_tmp = labels.cell_idx[labels.cell_idx.isin(cell_ids)]
        mappings_tmp = mappings.loc[mappings.cell_idx.isin(labels_tmp)]
    
        ax[i].scatter(mappings_tmp.y, mappings_tmp.z, c='gray', s=16, alpha=0.05)
        for k, gene in enumerate(genes):
            protein_tmp = mappings_tmp[mappings_tmp.gene_names==gene]
            ax[i].scatter(protein_tmp.y, protein_tmp.z, c=color_list[k],label=gene, s=16, alpha=1)
        ax[i].set_title(f'bin {i}')
    plt.legend()
    plt.savefig(f"{save_dir}/umap_{len(genes)}genes_bins.png", dpi=300)
    plt.close()
