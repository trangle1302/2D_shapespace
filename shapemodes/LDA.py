import sys
sys.path.append("..")
import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from s3_calculate_shapemodes_all import load_fft
import configs.config as cfg
import matplotlib.pyplot as plt

if __name__ == "__main__":
    project_dir = f"{cfg.PROJECT_DIR}" # /data/2Dshapespace/S-BIAD34
    load_raw_fft = False
    load_reduced_fft = True

    if load_raw_fft == True:
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

        df.index = df.matchid
        df = df.reindex(overlap_ids)
        df = df.drop('matchid', axis=1)
        assert (df.index == sc_stats.index).all()
        df = pd.concat(
                        [
                            pd.DataFrame(np.matrix(df).real),
                            pd.DataFrame(np.matrix(df).imag),
                        ],
                        axis=1,
                    )

        X_train, X_test, y_train, y_test = train_test_split(df, sc_stats['GMM_cc'], 
                                                            test_size=0.2, random_state=42, stratify=sc_stats['GMM_cc'])
    elif load_reduced_fft == True:
        df = pd.read_csv(f"{cfg.PROJECT_DIR}/shapemode/fft_cell_major_axis_polarized_cell_nuclei/transformed_matrix.csv")
        #df = df.drop('Unnamed: 0', axis=1)
        X_train, X_test, y_train, y_test = train_test_split(df[[f'PC{n}' for n in range(1,100,1)]], df['GMM_cc'], 
                                                            test_size=0.2, random_state=42, stratify=df['GMM_cc'])
    else:
        print('Please set load_raw_fft or load_reduced_fft to True')
    
    if False: #split
        X_train, X_test, y_train, y_test = train_test_split(df[['nu_area','cell_area']], df['GMM_cc'], 
                            test_size=0.1, random_state=42, stratify=df['GMM_cc'])

        clf = LinearDiscriminantAnalysis()
        clf.fit(X_train, y_train)
        pred = clf.predict(X_test)
        acc = accuracy_score(y_test, pred)
        cm = confusion_matrix(y_test, pred)
        print(f'Test set acc {acc}, confusion matrix: {cm}')
    else:
        X_train = df[[f'PC{n}' for n in range(1,101,1)]]
        y_train = df['GMM_cc']
        clf = LinearDiscriminantAnalysis()
        clf.fit(X_train, y_train)
        pred = clf.predict(X_train)
        acc = accuracy_score(y_train, pred)
        cm = confusion_matrix(y_train, pred)
        print(f'Train set acc {acc}, confusion matrix: {cm}')
        df_trans = clf.transform(X_train)

        d = {0 : 'r', 1:'b', 2:'g'}
        c = [d[i]  for i in y_train]
        plt.scatter(df_trans[:,0],df_trans[:,1], c=c)
        plt.savefig('lda.png')
