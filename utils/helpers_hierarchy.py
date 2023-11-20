import numpy as np
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as shc
import gseapy

def get_cluster_members(linkage_matrix, cluster_index):
    """ Get lowest level members of certain cluster based on linkage matrix
    
    Parameters
    ----------
    linkage_matrix : (n-1)x4 array of float
        output of scipy.cluster.hierarchy.linkage
    cluster_index : int
        cluster index to be examined
    Returns
    -------
    out : [] list
        list of idx of origianl members
    """
    n = linkage_matrix.shape[0] + 1
    clusters = {i: [i] for i in range(n)}

    for i, row in enumerate(linkage_matrix):
        cluster_id = i + n
        clusters[cluster_id] = clusters[row[0]] + clusters[row[1]]
        
    clusters_to_process = [cluster_index]

    original_members = []

    while len(clusters_to_process) > 0:
        current_cluster = clusters_to_process.pop()
        if current_cluster >= n:
            clusters_to_process.extend(clusters[current_cluster])
        else:
            original_members += clusters[current_cluster]
    assert len(original_members) == linkage_matrix[cluster_index-n,3]
    return original_members


def plot_dendrogram(clusters, save_path=None):
    fig, ax = plt.subplots(figsize=(20,20))
    n = clusters.shape[0]
    dn = shc.dendrogram(Z=clusters, orientation='right') # labels = features_tmp.gene_names.values,

    # plt.figure(figsize=(40, 40))
    # dn = shc.dendrogram(clusters, truncate_mode='lastp', distance_sort='ascending', orientation='right')
    ii = np.argsort(np.array(dn['dcoord'])[:, 1])
    for j, (icoord, dcoord) in enumerate(zip(dn['icoord'], dn['dcoord'])):
        x = 0.5 * sum(icoord[1:3])
        y = dcoord[1]
        ind = np.nonzero(ii == j)[0][0]
        ax.annotate(n+j, (y,x), va='top', ha='center')
    plt.tight_layout()
    if save_path!=None:
        plt.savefig(save_path)

# gseapy.get_library_name()
def gseapy_clusters(linkage_matrix, gene_names, cutting_thresholds=[2,10,90,200], save_path="./covar/PC1_b0_"):
    """ Enrichment analysis of all clusters when hierarhy is cut at certain thresholds
    
    Parameters
    ----------
    linkage_matrix : (n-1)x4 array of float
        output of scipy.cluster.hierarchy.linkage
    gene_names : list of str
        gene name list in the same order as clusters index
    cutting_thresholds: list of int
        distance threshold to make the cut in dendrogram
    save_path: str
        path to save outputs of geaspy.enr
    Returns
    -------
    None
        results saved in save_path
    """
    for threshold in cutting_thresholds:
        idx = shc.fcluster(linkage_matrix, threshold, "distance")
        print(f"Cluster flatten to {len(np.unique(idx))} clusters")
        for group_id in np.unique(idx)[:3]:
            gene_list = gene_names[idx==group_id]
            gene_list = [g.split(",")[0] for g in gene_list]
            try:
                enr = gseapy.enrichr(gene_list=gene_list,
                        gene_sets= [
                            'GO_Biological_Process_2023',
                            'GO_Cellular_Component_2023',
                            'GO_Molecular_Function_2023',
                            'Jensen_COMPARTMENTS',
                            'Reactome_2022',
                            #'KEGG_2021_Human',
                            'CORUM'
                        ],
                        #outdir=f"{save_path}_{threshold}_{group_id}",
                        cutoff=0.1, # Only affects the output figure, not the final output file
                        #format="pdf",
                    )
                tmp = enr.results
                tmp = tmp[tmp['Adjusted P-value']<=0.1]
                terms = '|'.join(list(tmp.Term.values))
                print(f"Distance threshold {threshold}, cluster {group_id}, n_members {len(gene_list)}, highest enrichr: {terms}")
            except:
                print(f"Distance threshold {threshold}, cluster {group_id}, n_members {len(gene_list)}, enrichr fails")
        print(f"##################### Moving up the hierarchy")

def factorize_into_quantiles(data, column_name, n):
    # Calculate quantiles
    quantiles = [data[column_name].quantile(q) for q in [i / n for i in range(1, n)]]
    # Create an empty list to store the lists for each quantile
    quantile_lists = []
    data['groups'] = 0
    # Iterate through the quantiles and create lists for each quantile
    for i in range(n):
        if i == 0:
            quantile_lists.append(list(data['Unnamed: 0'][data[column_name] <= quantiles[i]]))
            data.loc[data[column_name] <= quantiles[i],'groups'] = i
        elif i == n - 1:
            quantile_lists.append(list(data['Unnamed: 0'][data[column_name] > quantiles[i - 1]]))
            data.loc[data[column_name] > quantiles[i - 1],'groups'] = i
        else:
            quantile_lists.append(list(data['Unnamed: 0'][(data[column_name] > quantiles[i - 1]) & (data[column_name] <= quantiles[i])]))
            data.loc[(data[column_name] > quantiles[i - 1]) & (data[column_name] <= quantiles[i]),'groups'] = i
    return quantile_lists, quantiles, data


