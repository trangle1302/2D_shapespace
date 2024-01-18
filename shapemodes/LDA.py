import sys
sys.path.append("..")
import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from s3_calculate_shapemodes_all import load_fft
import configs.config as cfg

if __name__ == "__main__":
    project_dir = f"{cfg.PROJECT_DIR}" # /data/2Dshapespace/S-BIAD34
    lines = load_fft(cfg, project_dir)
    df = pd.DataFrame(lines).transpose()
    print(df.shape)
    if cfg.COEF_FUNC == "fft":
        df = df.applymap(lambda s: complex(s.replace("i", "j")))

    if cfg.MODE == "nuclei":
        df = df.iloc[:, (df.shape[1] // 2) :]
    elif cfg.MODE == "cell":
        df = df.iloc[:, : (df.shape[1] // 2)]

    df["matchid"] = [
        k.replace("/data/2Dshapespace/S-BIAD34/cell_masks/", "").replace(
            ".npy", ""
        )
        for k in df.index
    ]

    sc_stats = pd.read_csv(f"{cfg.PROJECT_DIR}/single_cell_statistics.csv")
    sc_stats["matchid"] = sc_stats.ab_id + "/" + sc_stats.cell_id
    print(f'Intersection: {sc_stats.matchid.isin(df.matchid).sum()}')
    sc_stats.index = sc_stats.matchid
    overlap_ids = sc_stats.matchid[sc_stats.matchid.isin(df.matchid)]
    sc_stats = sc_stats.reindex(overlap_ids)
    df_ = df.reindex(overlap_ids)
    #unmatched_idx = np.argwhere((df.matchid.values == sc_stats.matchid.values).astype('int') == 0)
    X_train, X_test, y_train, y_test = train_test_split(df_.drop('matchid', axis=1), sc_stats['GMM_cc'], test_size=0.33, random_state=42, stratify=
    sc_stats = pd.read_csv(f"{cfg.PROJECT_DIR}/single_cell_statistics.csv")
    sc_stats["matchid"] = sc_stats.ab_id + "/" + sc_stats.cell_id
    print(f'Intersection: {sc_stats.matchid.isin(df.matchid).sum()}')
    sc_stats.index = sc_stats.matchid
    overlap_ids = sc_stats.matchid[sc_stats.matchid.isin(df.matchid)]
    sc_stats = sc_stats.reindex(overlap_ids)
    df_ = df.copy()
    df_.index = df_.matchid
    df_ = df_.reindex(overlap_ids)
    df_ = df_.drop('matchid', axis=1)
    assert (df_.index == sc_stats.index).all()
    df_ = pd.concat(
                    [
                        pd.DataFrame(np.matrix(df_).real),
                        pd.DataFrame(np.matrix(df_).imag),
                    ],
                    axis=1,
                )
    #unmatched_idx = np.argwhere((df.matchid.values == sc_stats.matchid.values).astype('int') == 0)
    X_train, X_test, y_train, y_test = train_test_split(df_, sc_stats['GMM_cc'], 
                                                        test_size=0.33, random_state=42, stratify=sc_stats['GMM_cc'])
    clf = LinearDiscriminantAnalysis()
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    acc = accuracy_score(y_test, pred)
    cm = confusion_matrix(y_test, pred)
    print(f'Test set acc {acc}, confusion matrix: {cm}')
    df_trans = clf.transform(df)