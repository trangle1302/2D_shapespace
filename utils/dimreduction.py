import numpy as np
import pandas as pd


class ComplexPCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.components_ = self.u = self.s = self.principle_axis_ = None
        self.mean_ = None

    @property
    def explained_variance_ratio_(self):
        exp_var = np.round(self.s / np.sum(self.s), decimals=3)
        return exp_var

    @property
    def explained_variance_(self):
        return self.s

    def fit(self, matrix):
        n_samples, n_features = matrix.shape
        self.mean_ = matrix.mean(axis=0)
        center = matrix - self.mean_

        # stds, pcs = np.linalg.eigh(np.cov(center))
        # _, stds, pcs = np.linalg.svd(center)#/np.sqrt(matrix.shape[0]))

        u, s, vh = np.linalg.svd(
            center, full_matrices=True
        )  # full=False ==> num_pc = min(N, M)

        self.principle_axis_ = (
            vh  # already conjugated (reverted +/- sign on all imag components)
        )
        # Leave those components as rows of matrix so that it is compatible with Sklearn PCA.
        self.s = s  # lapack svd returns eigenvalues with s ** 2 sorted descending
        self.u = u
        self.components_ = vh
        
    def transform(self, matrix):
        """In SVD:
        X=USV⊤
        US : principle components
        V⊤ : principle axis
        λi=si**2/(n−1) : Eigenvalues λi show variances of the respective PCs
        """
        data = matrix - self.mean_
        result = data @ self.principle_axis_.T  # US
        return result

    def inverse_transform(self, matrix):
        result = matrix @ np.conj(self.principle_axis_)
        return self.mean_ + result


"""
def ComplexPCA():
    def __init__(self, n_components):
        self.n_components = n_components
        self.mean_ = None
        self.s = None
        
    @property
    def explained_variance_(self):
        return self.s
    
    def fit_transform(self, X):
        covariance_matrix = np.cov(X.T)
        eigen_values, eigen_vectors = np.linalg.eig(covariance_matrix)
        eigen_values, eigen_vectors = np.linalg.eigh(covariance_matrix)
        
        variance_explained = []
        for i in eigen_values:
             variance_explained.append((i/sum(eigen_values))*100)
        self.s = variance_explained
"""


class ComplexScaler:
    def __init__(self):
        self.mean_ = None
        self.std_ = None

    def fit(self, matrix, axis=0):
        self.mean_ = matrix.mean(axis=axis)
        self.std_ = matrix.std(axis=axis)
        matrix = np.asmatrix(matrix)
        # matrix has shape of (n_sample, n_coeff)
        """
        std = []
        for col in matrix.T:
            real = [x.real for x in col]
            imag = [x.imag for x in col]
            std += [complex(np.std(real), np.std(imag))]
        self.std_ = std
        """

    def transform(self, matrix):
        data = matrix - self.mean_
        result = data / self.std_
        return result

    def inverse_transform(self, matrix):
        result = matrix * self.std_
        return self.mean_ + result


class ComplexNormalizer:
    def __init__(self):
        self.max_ = None
        self.n_coef = None

    """
    def fit(self, matrix, axis=0):
        # matrix has shape of (n_sample, n_coeff)
        max_r = []
        max_i = []
        for pc in range(matrix.shape[1]):
            col = matrix[pc]
            real = [abs(x.real) for x in col]
            imag = [abs(x.imag) for x in col]
            max_r += [np.max(real)]
            max_i += [np.max(imag)]
        self.max_r_ = max_r
        self.max_i_ = max_i
    """

    def fit(self, matrix, axis=0):
        n = matrix.shape[1] // 2
        self.n_coef = n
        # matrix has shape of (n_sample, n_coeff)
        self.max_ = [x.real for x in abs(matrix).max(axis=0)[0:n]]
        """
        max_r = []
        max_i = []
        for k in range(n):
            col = matrix[n+k] # max of nucleus
            real = [abs(x.real) for x in col]
            imag = [abs(x.imag) for x in col]
            max_r += [np.max(real)]
            max_i += [np.max(imag)]
        self.max_r_ = max_r
        self.max_i_ = max_i
        """

    def transform(self, matrix):
        try:
            n = matrix.shape[1] // 2
        except:
            n = len(matrix) // 2
        # assert self.n_coef == n
        # matrix has shape of (n_sample, n_coeff)
        for k in range(n):
            for pc in [k, n + k]:
                col = matrix[pc].copy() / self.max_[k]
                matrix[pc] = col
                """
                real = [x.real for x in col]
                imag = [x.imag for x in col]
                matrix[pc] = [
                    complex(r / self.max_r_[k], i / self.max_i_[k])
                    for r, i in zip(real, imag)
                ]
                """

        result = matrix
        return result

    def inverse_transform(self, matrix):
        try:
            n = matrix.shape[1] // 2
        except:
            n = len(matrix) // 2
        # assert self.n_coef == n

        for k in range(n):
            for pc in [k, n + k]:
                col = matrix[pc].copy() * self.max_[k]
                matrix[pc] = col
                """
                real = [x.real for x in col]
                imag = [x.imag for x in col]
                matrix[pc] = [
                    complex(r * self.max_r_[k], i * self.max_i_[k])
                    for r, i in zip(real, imag)
                ]
                """
        result = matrix
        return result


class LogScaler:
    def __init__(self):
        pass

    def fit(self, matrix):
        pass

    def transform(self, matrix):
        data = np.log(matrix)
        return data

    def inverse_transform(self, matrix):
        result = np.exp(matrix)
        return result


def calculate_feature_importance(pca, df_trans):
    df_dimred = {}
    loading = pca.components_.T * np.sqrt(pca.explained_variance_)
    for comp, pc_name in enumerate(df_trans.columns):
        load = loading[:, comp]
        pc = [v for v in load]
        apc = [v for v in np.abs(load)]
        total = np.sum(apc)
        cpc = [100 * v / total for v in apc]
        df_dimred[pc_name] = pc
        df_dimred[pc_name.replace("_PC", "_aPC")] = apc
        df_dimred[pc_name.replace("_PC", "_cPC")] = cpc
    df_dimred["features"] = df_trans.columns
    df_dimred = pd.DataFrame(df_dimred)
    df_dimred = df_dimred.set_index("features", drop=True)
    return df_dimred
