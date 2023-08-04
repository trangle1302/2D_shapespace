import sys
sys.path.append("..")
import pandas as pd
from scipy.stats import skewtest
import configs.config as cfg

def skew_test(distr1):
    zscore, pval = skewtest(distr1, axis=0)
    return pval

def main():
    df = pd.read_csv(f"{cfg.PROJECT_DIR}/shapemode/{cfg.ALIGNMENT}_{cfg.MODE}/transformed_matrix.scv")
    mappings = pd.read_csv(f"{cfg.META_PATH}")
    for df_ in mappings.groupby("ensembl_ids"):
        df_ = 

if __name__ == "__main__":
    main()