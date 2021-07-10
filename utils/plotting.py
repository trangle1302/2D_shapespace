import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils.helpers import equidistance
from utils.coefs import inverse_fft
from sklearn.linear_model import LinearRegression
from matplotlib.animation import FuncAnimation, PillowWriter


class PlotShapeModes:
    def __init__(
        self, pca, features_transform, n_coef, pc_keep, scaler=None, complex_type=True
    ):
        self.pca = pca
        self.sc = scaler
        self.matrix = features_transform
        self.n = n_coef
        self.pc_keep = pc_keep
        self.complex = complex_type
        self.midpoints = None
        self.std = None

        # mean = abs(self.matrix).mean(axis=0)
        # self.midpoints = mean
        self.std = self.matrix.std()

        mean = []
        # std = []
        for c in self.matrix:

            col = self.matrix[c]
            real_ = [x.real for x in col]
            real = real_  # [-abs(x) for x in real_]
            # p = np.percentile(real_, [5, 95])
            # real = [-abs(r) for r in real_ if p[0] <= r <= p[1]]

            imag_ = [x.imag for x in col]
            imag = imag_  # [-abs(x) for x in imag_]
            # p = np.percentile(imag_, [5, 95])
            # imag = [i for i in imag_ if p[0] <= i <= p[1]]
            # std += [complex(np.std(real), np.std(imag))]
            mean += [complex(np.mean(real), np.mean(imag))]
            """
            col = self.matrix[c]
            p = np.percentile(col, [5, 95])
            col = [x for x in col if p[0] <= x <= p[1]]
            std += [np.std(col)]
            mean += [np.mean(col)]
            """
        self.midpoints = pd.Series(mean, index=self.matrix.columns)
        # self.std = pd.Series(std, index=self.matrix.columns)

        self.equipoints = None
        # self.get_equipoints()
        self.stdpoints = None
        self.get_z()
        self.lmpoints = None
        # self.get_lm()

    def plot_pc_dist(self, pc_name):
        cnums = self.matrix[pc_name]
        X = [x.real for x in cnums]
        Y = [x.imag for x in cnums]
        midpoint = self.midpoints[pc_name]

        plt.scatter(X, Y, color="blue", alpha=0.1)
        plt.scatter(midpoint.real, midpoint.imag, label="star", marker="*", color="red")
        # for p in self.lmpoints[pc_name]:
        for p in self.stdpoints[pc_name]:
            # for p in self.equipoints[pc_name]:
            plt.scatter(p.real, p.imag, label="star", marker="*", color="orange")

        plt.xlabel("real axis")
        plt.ylabel("imaginary axis")
        plt.title(pc_name)
        plt.show()

    def plot_pc_hist(self, pc_name):
        cnums = self.matrix[pc_name]
        X = [x.real for x in cnums]

        plt.hist(X, color="blue", alpha=0.1)
        # for p in self.lmpoints[pc_name]:
        for p in self.stdpoints[pc_name]:
            plt.scatter(p.real, 100, label="star", marker="*", color="orange")
        plt.ylabel("density")
        plt.xlabel("real axis")
        plt.title(pc_name)
        plt.show()

    def plot_avg_cell(self):
        midpoint = self.midpoints.copy()
        fcoef = self.pca.inverse_transform(midpoint)
        if not self.complex:
            real = fcoef[: len(fcoef) // 2]
            imag = fcoef[len(fcoef) // 2 :]
            fcoef = [complex(r, i) for r, i in zip(real, imag)]
        if self.sc != None:
            fcoef = self.sc.inverse_transform(fcoef)
        fcoef_c = fcoef[0 : self.n * 2]
        fcoef_n = fcoef[self.n * 2 :]
        ix_n, iy_n = inverse_fft(fcoef_n[0 : self.n], fcoef_n[self.n :])
        ix_c, iy_c = inverse_fft(fcoef_c[0 : self.n], fcoef_c[self.n :])

        ix_n, iy_n = equidistance(ix_n.real, iy_n.real, self.n * 10)
        ix_c, iy_c = equidistance(ix_c.real, iy_c.real, self.n * 10)
        plt.title("Avg cell")
        plt.plot(ix_n, iy_n)
        plt.plot(ix_c, iy_c)
        plt.axis("scaled")

    def get_equipoints(self):
        points = dict()
        for c in self.pc_keep:
            col = self.matrix[c]
            real = [x.real for x in col]
            imag = [x.imag for x in col]
            r_, i_ = equidistance(real, imag, 9)
            points[c] = [complex(r, i) for r, i in zip(r_, i_)]
        self.equipoints = points

    def get_z(self):
        points = dict()
        for c in self.pc_keep:
            midpoint = self.midpoints[c].copy()
            std_ = self.std[c].copy()
            p_std = []
            for k in np.arange(-1.5, 1.5, 0.3):
                p_std += [midpoint + k * std_]
            points[c] = p_std
        self.stdpoints = points

    def get_lm(self):
        points = dict()
        for c in self.pc_keep:
            midpoint = self.midpoints[c].copy()
            std_ = self.std[c].copy()
            col = self.matrix[c]
            real = [x.real for x in col]
            imag = [x.imag for x in col]
            X = np.array(real).reshape((-1, 1))
            Y = np.array(imag)
            reg = LinearRegression().fit(X, Y)
            p_r = [midpoint.real + k * std_.real for k in np.arange(-2.5, 2.5, 0.5)]
            p_r = np.array(p_r).reshape((-1, 1))
            p_i = reg.predict(p_r)
            points[c] = [complex(p_r[k], p_i[k]) for k in range(len(p_r))]
        self.lmpoints = points

    def plot_shape_variation(self, pc_name):
        fig, ax = plt.subplots(1, len(self.stdpoints[pc_name]), figsize=(15, 4))
        for i, p in enumerate(self.stdpoints[pc_name]):
            # for i, p in enumerate(self.equipoints[pc_name]):
            # for i, p in enumerate(self.lmpoints[pc_name]):
            cell_coef = self.midpoints.copy()
            cell_coef[pc_name] = p
            fcoef = self.pca.inverse_transform(cell_coef)
            if self.sc != None:
                fcoef = self.sc.inverse_transform(fcoef)
            if not self.complex:
                real = fcoef[: len(fcoef) // 2]
                imag = fcoef[len(fcoef) // 2 :]
                fcoef = [complex(r, i) for r, i in zip(real, imag)]
            fcoef_c = fcoef[0 : self.n * 2]
            fcoef_n = fcoef[self.n * 2 :]
            ix_n, iy_n = inverse_fft(fcoef_n[0 : self.n], fcoef_n[self.n :])
            ix_c, iy_c = inverse_fft(fcoef_c[0 : self.n], fcoef_c[self.n :])

            # ix_n, iy_n = inverse_fft(fcoef[0:self.n], fcoef[2*self.n:3*self.n])
            # ix_c, iy_c = inverse_fft(fcoef[self.n:2*self.n], fcoef[3*self.n:])

            # ax[i].title(f'Cell at {}std')
            ax[i].plot(ix_n.real, iy_n.real)
            ax[i].plot(ix_c.real, iy_c.real)
            ax[i].axis("scaled")
        plt.show()

    def plot_shape_variation_gif(self, pc_name):
        def init():
            """Local function to init space in animated plots"""
            ax.set_xlim(-600, 600)
            ax.set_ylim(-600, 600)

        def update(p):
            cell_coef = self.midpoints.copy()
            cell_coef[pc_name] = p
            fcoef = self.pca.inverse_transform(cell_coef)
            if self.sc != None:
                fcoef = self.sc.inverse_transform(fcoef)
            if not self.complex:
                real = fcoef[: len(fcoef) // 2]
                imag = fcoef[len(fcoef) // 2 :]
                fcoef = [complex(r, i) for r, i in zip(real, imag)]
            fcoef_c = fcoef[0 : self.n * 2]
            fcoef_n = fcoef[self.n * 2 :]
            ix_n, iy_n = inverse_fft(fcoef_n[0 : self.n], fcoef_n[self.n :])
            ix_c, iy_c = inverse_fft(fcoef_c[0 : self.n], fcoef_c[self.n :])

            nu.set_data(ix_n.real, iy_n.real)
            cell.set_data(ix_c.real, iy_c.real)

        fig, ax = plt.subplots()
        fig.suptitle(pc_name)
        (nu,) = plt.plot([], [], "b", lw=2)
        (cell,) = plt.plot([], [], "m", lw=2)
        ani = FuncAnimation(
            fig,
            update,
            # self.lmpoints[pc_name] + self.lmpoints[pc_name][::-1],
            self.stdpoints[pc_name] + self.stdpoints[pc_name][::-1],
            init_func=init,
        )
        ax.axis("scaled")
        writer = PillowWriter(fps=5)
        ani.save(
            f"C:/Users/trang.le/Desktop/2D_shape_space/tmp/shapevar_{pc_name}.gif",
            writer=writer,
        )


def display_scree_plot(pca):
    """Display a scree plot for the pca"""

    scree = pca.explained_variance_ratio_ * 100
    plt.bar(np.arange(len(scree)) + 1, scree)
    plt.plot(np.arange(len(scree)) + 1, scree.cumsum(), c="red")

    for thres in [70, 80, 90]:
        idx = np.searchsorted(scree.cumsum(), thres)
        plt.plot(idx + 1, scree.cumsum()[idx], c="red", marker="o")
        plt.annotate(f"{idx} PCs", xy=(idx + 3, scree.cumsum()[idx] - 5))
    plt.xlabel("Number of PCs")
    plt.ylabel("Percentage explained variance")
    plt.title("Scree plot")
    # plt.hlines(y=70, xmin = 0, xmax = len(scree), linestyles='dashed', alpha=0.5)
    # plt.vlines(x=np.argmax(scree.cumsum()>70), ymin = 0, ymax = 100, linestyles='dashed', alpha=0.5)
    # plt.hlines(y=80, xmin = 0, xmax = len(scree), linestyles='dashed', alpha=0.5)
    # plt.vlines(x=np.argmax(scree.cumsum()>80), ymin = 0, ymax = 100, linestyles='dashed', alpha=0.5)
    plt.show(block=False)
