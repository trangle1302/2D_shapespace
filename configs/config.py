LABEL_TO_ALIAS = {
    0: "Nucleoplasm",
    1: "NuclearM",
    2: "Nucleoli",
    3: "NucleoliFC",
    4: "NuclearS",
    5: "NuclearB",
    6: "EndoplasmicR",
    7: "GolgiA",
    8: "IntermediateF",
    9: "ActinF",
    10: "Microtubules",
    11: "MitoticS",
    12: "Centrosome",
    13: "PlasmaM",
    14: "Mitochondria",
    15: "Aggresome",
    16: "Cytosol",
    17: "VesiclesPCP",
    19: "Negative",
    19: "Multi-Location",
}

COLORS = [
    "#f44336",
    "#e91e63",
    "#9c27b0",
    "#673ab7",
    "#3f51b5",
    "#2196f3",
    "#03a9f4",
    "#00bcd4",
    "#009688",
    "#4caf50",
    "#8bc34a",
    "#cddc39",
    "#ffeb3b",
    "#ffc107",
    "#ff9800",
    "#ff5722",
    "#795548",
    "#9e9e9e",
    "#607d8b",
    "#dddddd",
    "#212121",
    "#ff9e80",
    "#ff6d00",
    "#ffff00",
    "#76ff03",
    "#00e676",
    "#64ffda",
    "#18ffff",
]

COLORS_MAP = {
    "Nucleoplasm": "Blues",
    "GolgiA": "Greens",
    "IntermediateF": "Oranges",
    "Mitochondria": "Reds",
    "PlasmaM": "Purples",
    "Cytosol": "Greys",
}

ORGANELLES = ["Nucleoplasm","Nucleoli","NucleoliFC","NuclearS","NuclearB","NuclearM",
              "GolgiA", "IntermediateF","ActinF","Mitochondria",""
              "Cytosol","PlasmaM","Microtubules"]

# >>>>>>>>>>>>>>>>>>>>> PARAM CONFIGS
CELL_LINE = "RT4"
N_COEFS = 128
N_SAMPLES = -1  # 5000
N_CV = 1
MODE = "cell_nuclei"  # ["nuclei" #"cell_nuclei"]
ALIGNMENT = "fft_cell_major_axis_polarized"  # ["fft_nuclei_major_axis","fft_nuclei_major_axis","fft_cell_nuclei_centroid"]
COEF_FUNC = "fft" # ["efd", "wavelet"]
N_ISOS = [10,20]

# >>>>>>>>>>>>>>>>>>>>> COMPUTE RESOURCE + PACKAGE
SERVER = "sherlock"
if SERVER == "callisto":
    PROJECT_DIR = f"/data/2Dshapespace/{CELL_LINE.replace(' ','_')}"
    META_PATH = "/data/kaggle-dataset/publicHPA_umap/results/webapp/sl_pHPA_15_0.05_euclidean_100000_rmoutliers_ilsc_3d_bbox_rm_border.csv"
elif SERVER == "sherlock":
    PROJECT_DIR = f"/scratch/groups/emmalu/2Dshapespace/{CELL_LINE.replace(' ','_')}"
    META_PATH = "/scratch/groups/emmalu/sl_pHPA_15_0.05_euclidean_100000_rmoutliers_ilsc_3d_bbox_rm_border.csv"
