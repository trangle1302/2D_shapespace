import sys
sys.path.append("..")
import numpy as np
import matplotlib.pyplot as plt
from utils.helpers import equidistance, get_line
from coefficients import coefs
from sklearn.linear_model import LinearRegression
from matplotlib.animation import FuncAnimation, PillowWriter
from imageio import imread
from scipy.ndimage import rotate
from more_itertools import windowed
from skimage import exposure
from skimage.filters import threshold_mean
from warps import parameterize

class PlotShapeModes:
    def __init__(
        self,
        pca,
        features_transform,
        n_coef,
        pc_keep,
        scaler=None,
        complex_type=True,
        fourier_algo = 'fft',
        inverse_func=coefs.inverse_wavelet,
        mode = "cell_nuclei"
    ):
        self.pca = pca
        self.sc = scaler
        self.matrix = features_transform
        self.n = n_coef
        self.pc_keep = pc_keep
        self.complex = complex_type
        self.fourier_algo = fourier_algo
        self.inverse_func = inverse_func
        self.midpoints = None
        self.std = None
        self.protein_intensities = None
        self.percent_var = np.round(pca.explained_variance_ratio_ * 100,2)
        self.mode = mode # "cell_nuclei", "nuclei" ("cell" option has not been implemented)

        mean = self.matrix.mean()  # .clip(0, None).mean()
        # mean = abs(self.matrix).mean(axis=0)
        self.midpoints = mean
        self.std = self.matrix.std()
        """
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
            '''
            col = self.matrix[c]
            p = np.percentile(col, [5, 95])
            col = [x for x in col if p[0] <= x <= p[1]]
            std += [np.std(col)]
            mean += [np.mean(col)]
            '''
        self.midpoints = pd.Series(mean, index=self.matrix.columns)
        # self.std = pd.Series(std, index=self.matrix.columns)
        """
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

    def plot_avg_cell(self, dark=True, save_dir="C:/Users/trang.le/Desktop/2D_shape_space/shapespace_plots"):
        midpoint = self.midpoints.copy()
        fcoef = self.pca.inverse_transform(midpoint)
        if self.fourier_algo == "fft":
            if not self.complex:
                real = fcoef[: len(fcoef) // 2]
                imag = fcoef[len(fcoef) // 2 :]
                fcoef = [complex(r, i) for r, i in zip(real, imag)]
            if self.sc != None:
                fcoef = self.sc.inverse_transform(fcoef)
            fcoef_c = fcoef[0 : self.n * 2]
            fcoef_n = fcoef[self.n * 2 :]        
            if True: # nu_coef*2 so need to divide by 2
                fcoef_n = [x_/4 for x_ in fcoef_n]
            ix_n, iy_n = self.inverse_func(fcoef_n[0 : self.n], fcoef_n[self.n :])
            ix_c, iy_c = self.inverse_func(fcoef_c[0 : self.n], fcoef_c[self.n :])

        elif self.fourier_algo == "efd":
            fcoef_c = fcoef[: len(fcoef) // 2]
            fcoef_n = fcoef[len(fcoef) // 2 :]
            ix_n, iy_n = self.inverse_func(fcoef_n, n_points=self.n*2)
            ix_c, iy_c = self.inverse_func(fcoef_c, n_points=self.n*2)

        if dark:
            plt.style.use('dark_background')
            plt.rcParams['savefig.facecolor'] = '#191919'
            plt.rcParams['figure.facecolor'] ='#191919'
            plt.rcParams['axes.facecolor'] ='#191919'
        else:
            plt.style.use('default')
        np.savez(f"{save_dir}/Avg_cell.npz", ix_n=ix_n.real, iy_n=iy_n.real, ix_c=ix_c.real, iy_c=iy_c.real)

        ix_n, iy_n = equidistance(ix_n.real, iy_n.real, self.n * 10)
        ix_c, iy_c = equidistance(ix_c.real, iy_c.real, self.n * 10)
        plt.title("Avg cell")
        plt.plot(ix_n, iy_n,"#8ab0cf")
        plt.plot(ix_c, iy_c,"m")
        plt.axis("scaled")
        plt.savefig(f"{save_dir}/Avg_cell.jpg", bbox_inches="tight")
        plt.close()


    def plot_avg_nucleus(self, dark=True, save_dir="C:/Users/trang.le/Desktop/2D_shape_space/shapespace_plots"):
        midpoint = self.midpoints.copy()
        fcoef = self.pca.inverse_transform(midpoint)
        if self.fourier_algo == "fft":
            if not self.complex:
                real = fcoef[: len(fcoef) // 2]
                imag = fcoef[len(fcoef) // 2 :]
                fcoef = [complex(r, i) for r, i in zip(real, imag)]
            if self.sc != None:
                fcoef = self.sc.inverse_transform(fcoef)
            if len(fcoef) == self.n*2: # If there's equal number of column as n_coefs needed to reconstruct a shape
                fcoef_n = fcoef
            else: # else: each row is in the format of [fcoef_c, fcoef_n]
                fcoef_n = fcoef[self.n * 2 :] 
            ix_n, iy_n = self.inverse_func(fcoef_n[0 : self.n], fcoef_n[self.n :])
    
        elif self.fourier_algo == "efd":
            raise NotImplementedError

        if dark:
            plt.style.use('dark_background')
            plt.rcParams['savefig.facecolor'] = '#191919'
            plt.rcParams['figure.facecolor'] ='#191919'
            plt.rcParams['axes.facecolor'] ='#191919'
        else:
            plt.style.use('default')
        np.savez(f"{save_dir}/Avg_nucleus.npz", ix_n=ix_n.real, iy_n=iy_n.real)

        ix_n, iy_n = equidistance(ix_n.real, iy_n.real, self.n * 10)
        plt.title("Average nucleus")
        plt.plot(ix_n, iy_n,"#8ab0cf")
        plt.axis("scaled")
        plt.savefig(f"{save_dir}/Avg_nucleus.jpg", bbox_inches="tight")
        plt.close()
        
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
            for k in np.linspace(-1.5,1.5,7): # control variation range with np.linspace(-2,2,9) or np.linspace(-1.5,1.5,11)
                p_std += [midpoint + k * std_]
            points[c] = p_std
        self.stdpoints = points
    
    def assign_cells(self, pc_name):
        cnums = self.matrix[pc_name]
        minnum = min(cnums)
        maxnum = max(cnums)
        
        ws = windowed(self.stdpoints[pc_name], 2)
        points = []
        for w in ws:
            points += [(w[0]+w[1])/2]
        points = [minnum] + points + [maxnum]
        ws = windowed(points, 2)
        idxes_assigned = []
        binned_links = []
        for w in ws:
            idxes = np.where((cnums>w[0]) & (cnums<w[1]))
            idxes_assigned += [idxes]
            binned_links += [cnums.index[idxes]]
        return idxes_assigned, binned_links
    
    
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

    def plot_shape_variation(self, pc_name, dark=True, save_dir="C:/Users/trang.le/Desktop/2D_shape_space/shapespace_plots"):
        if dark:
            plt.style.use('dark_background')
            plt.rcParams['savefig.facecolor'] = '#191919'
            plt.rcParams['figure.facecolor'] ='#191919'
            plt.rcParams['axes.facecolor'] = '#191919'
        else:
            plt.style.use('default')
        fig, ax = plt.subplots(1, len(self.stdpoints[pc_name]), figsize=(15, 4),sharex=True, sharey=True)
        if self.mode == "nuclei":
            nuc = []
        elif self.mode == "cell":
            mem = []
        elif self.mode == "cell_nuclei":
            nuc = []
            mem = []
        for i, p in enumerate(self.stdpoints[pc_name]):
            # for i, p in enumerate(self.equipoints[pc_name]):
            # for i, p in enumerate(self.lmpoints[pc_name]):
            cell_coef = self.midpoints.copy()
            cell_coef[pc_name] = p
            fcoef = self.pca.inverse_transform(cell_coef)
            if self.sc != None:
                fcoef = self.sc.inverse_transform(fcoef)
            if self.fourier_algo == "fft":
                if not self.complex:
                    real = fcoef[: len(fcoef) // 2]
                    imag = fcoef[len(fcoef) // 2 :]
                    fcoef = [complex(r, i) for r, i in zip(real, imag)]
                
                if self.mode == "nuclei":
                    if len(fcoef) == self.n*2: # If there's equal number of column as n_coefs needed to reconstruct a shape
                        fcoef_n = fcoef
                    else: # else: each row is in the format of [fcoef_c, fcoef_n]
                        fcoef_n = fcoef[self.n * 2 :] 
                    
                    if True: # nu_coef*2 so need to divide by 2
                        fcoef_n = [x_/4 for x_ in fcoef_n]

                    ix_n, iy_n = self.inverse_func(fcoef_n[0 : self.n], fcoef_n[self.n :])
                    nuc += [np.concatenate([ix_n.real, iy_n.real])]
                    ax[i].plot(ix_n.real, iy_n.real, "#8ab0cf")
                elif self.mode == "cell":                   
                    raise NotImplementedError
                elif self.mode == "cell_nuclei":                
                    fcoef_c = fcoef[0 : self.n * 2]
                    fcoef_n = fcoef[self.n * 2 :]
                                        
                    if True: # nu_coef*2 so need to divide by 2
                        fcoef_n = [x_/4 for x_ in fcoef_n]

                    ix_n, iy_n = self.inverse_func(fcoef_n[0 : self.n], fcoef_n[self.n :])
                    ix_c, iy_c = self.inverse_func(fcoef_c[0 : self.n], fcoef_c[self.n :])    
                    nuc += [np.concatenate([ix_n.real, iy_n.real])]
                    mem += [np.concatenate([ix_c.real, iy_c.real])]
                    ax[i].plot(ix_n.real, iy_n.real, "#8ab0cf")
                    ax[i].plot(ix_c.real, iy_c.real, "m")
            elif self.fourier_algo == "efd":
                fcoef_c = fcoef[: len(fcoef) // 2]
                fcoef_n = fcoef[len(fcoef) // 2 :]
                ix_n, iy_n = self.inverse_func(fcoef_n, n_points=self.n*2)
                ix_c, iy_c = self.inverse_func(fcoef_c, n_points=self.n*2)

                ax[i].plot(ix_n.real, iy_n.real, "#8ab0cf")
                ax[i].plot(ix_c.real, iy_c.real, "m")

            # ix_n, iy_n = self.inverse_fun(fcoef[0:self.n], fcoef[2*self.n:3*self.n])
            # ix_c, iy_c = self.inverse_fun(fcoef[self.n:2*self.n], fcoef[3*self.n:])
            ax[i].axis("scaled")
        pc_var = self.percent_var[int(list(filter(str.isdigit, pc_name))[0]) -1]
        plt.suptitle(f"{pc_name} - {pc_var}%", fontsize=16)
        plt.savefig(f"{save_dir}/shapevar_{pc_name}_{self.mode}.png", bbox_inches='tight', pad_inches=0)
        plt.show()
        plt.close()        
        
        if self.mode == "nuclei":
            np.savez(f"{save_dir}/shapevar_{pc_name}_{self.mode}.npz", 
                nuc=np.array(nuc))
        elif self.mode == "cell":
            np.savez(f"{save_dir}/shapevar_{pc_name}_{self.mode}.npz", 
                mem=np.array(mem))
        elif self.mode == "cell_nuclei":
            np.savez(f"{save_dir}/shapevar_{pc_name}_{self.mode}.npz", 
                nuc=np.array(nuc), 
                mem=np.array(mem))
        
    def plot_shape_variation_gif(self, pc_name, dark=True, save_dir=""):
        def init():
            """Local function to init space in animated plots"""
            ax.set_xlim(-600, 600)
            ax.set_ylim(-650, 600)

        def update(p):
            cell_coef = self.midpoints.copy()
            cell_coef[pc_name] = p
            fcoef = self.pca.inverse_transform(cell_coef)
            if self.sc != None:
                fcoef = self.sc.inverse_transform(fcoef)
            
            if self.fourier_algo == "fft":
                if not self.complex:
                    real = fcoef[: len(fcoef) // 2]
                    imag = fcoef[len(fcoef) // 2 :]
                    fcoef = [complex(r, i) for r, i in zip(real, imag)]
                if self.mode == "nuclei":
                    if len(fcoef) == self.n*2: # If there's equal number of column as n_coefs needed to reconstruct a shape
                        fcoef_n = fcoef
                    else: # else: each row is in the format of [fcoef_c, fcoef_n]
                        fcoef_n = fcoef[self.n * 2 :]  
                                            
                    if True: # nu_coef*2 so need to divide by 2
                        fcoef_n = [x_/4 for x_ in fcoef_n]

                    ix_n, iy_n = self.inverse_func(fcoef_n[0 : self.n], fcoef_n[self.n :])                      
                    nu.set_data(ix_n.real, iy_n.real)
                elif self.mode == "cell":
                    raise NotImplementedError   
                elif self.mode == "cell_nuclei":
                    fcoef_c = fcoef[0 : self.n * 2]
                    fcoef_n = fcoef[self.n * 2 :]
                    
                    if True: # nu_coef*2 so need to divide by 2
                        fcoef_n = [x_/4 for x_ in fcoef_n]
                        
                    ix_n, iy_n = self.inverse_func(fcoef_n[0 : self.n], fcoef_n[self.n :])
                    ix_c, iy_c = self.inverse_func(fcoef_c[0 : self.n], fcoef_c[self.n :])
                    
                    nu.set_data(ix_n.real, iy_n.real)
                    cell.set_data(ix_c.real, iy_c.real)
                
                else: 
                    raise NotImplementedError
                
            elif self.fourier_algo == "efd":
                if self.mode == "cell_nuclei":
                    fcoef_c = fcoef[: len(fcoef) // 2]
                    fcoef_n = fcoef[len(fcoef) // 2 :]
                    ix_n, iy_n = self.inverse_func(fcoef_n, n_points=self.n*2)
                    ix_c, iy_c = self.inverse_func(fcoef_c, n_points=self.n*2)
                    
                    nu.set_data(ix_n.real, iy_n.real)
                    cell.set_data(ix_c.real, iy_c.real)                
                else:
                    raise NotImplementedError

        if dark:
            plt.style.use('dark_background')
            plt.rcParams['savefig.facecolor'] = '#191919'
            plt.rcParams['figure.facecolor'] ='#191919'
            plt.rcParams['axes.facecolor'] = '#191919'
        else:
            plt.style.use('default')
        fig, ax = plt.subplots(figsize=(8,8))
        fig.suptitle(pc_name, y=0.95)
        if self.mode == "nuclei":
            (nu,) = plt.plot([], [], "#8ab0cf", lw=5)
        elif self.mode == "cell":
            (cell,) = plt.plot([], [], "m", lw=5)
        elif self.mode == "cell_nuclei":
            (nu,) = plt.plot([], [], "#8ab0cf", lw=5)
            (cell,) = plt.plot([], [], "m", lw=5)
        ax.axis("scaled")
        #ax.set_facecolor('#191919')        
        #fig.patch.set_facecolor('#191919')
        
        ani = FuncAnimation(
            fig,
            update,
            # self.lmpoints[pc_name] + self.lmpoints[pc_name][::-1],
            self.stdpoints[pc_name] + self.stdpoints[pc_name][::-1],
            init_func=init,
        )
        writer = PillowWriter(fps=5)
        ani.save(
            f"{save_dir}/shapevar_{pc_name}.gif",
            writer=writer,
        )
        plt.close()


    def plot_protein_through_shape_variation_gif(self, pc_name, title='', dark=True, point_size=4,save_dir="C:/Users/trang.le/Desktop/2D_shape_space/shapespace_plots"):
        def init():
            """Local function to init space in animated plots"""
            ax.set_xlim(-600, 600)
            ax.set_ylim(-650, 600)

        def update(p):
            i = np.where(self.stdpoints[pc_name] == p)[0][0]
            cell_coef = self.midpoints.copy()
            cell_coef[pc_name] = p
            fcoef = self.pca.inverse_transform(cell_coef)
            if self.sc != None:
                fcoef = self.sc.inverse_transform(fcoef)

            if self.fourier_algo == "fft":
                if not self.complex:
                    real = fcoef[: len(fcoef) // 2]
                    imag = fcoef[len(fcoef) // 2 :]
                    fcoef = [complex(r, i) for r, i in zip(real, imag)]
                fcoef_c = fcoef[0 : self.n * 2]
                fcoef_n = fcoef[self.n * 2 :]
                ix_n, iy_n = self.inverse_func(fcoef_n[0 : self.n], fcoef_n[self.n :])
                ix_c, iy_c = self.inverse_func(fcoef_c[0 : self.n], fcoef_c[self.n :])
            elif self.fourier_algo == "efd":
                fcoef_c = fcoef[: len(fcoef) // 2]
                fcoef_n = fcoef[len(fcoef) // 2 :]
                ix_n, iy_n = self.inverse_func(fcoef_n, n_points=self.n*2)
                ix_c, iy_c = self.inverse_func(fcoef_c, n_points=self.n*2)

            nu.set_data(ix_n.real, iy_n.real)
            cell.set_data(ix_c.real, iy_c.real)

            x_,y_ = parameterize.get_coordinates(
                np.concatenate([ix_n.real, iy_n.real]), 
                np.concatenate([ix_c.real, iy_c.real]), 
                [0,0], 
                n_isos = [10,10], 
                plot=False)
            
            np.savez(f"{save_dir}/{title}_{pc_name}.npz", x=x_, y=y_)
            
            ipoints0.set_offsets(np.c_[x_[0],y_[0]])
            ipoints0.set_array(self.protein_intensities[i][0])
            ipoints1.set_offsets(np.c_[x_[1],y_[1]])
            ipoints1.set_array(self.protein_intensities[i][1])
            ipoints2.set_offsets(np.c_[x_[2],y_[2]])
            ipoints2.set_array(self.protein_intensities[i][2])
            ipoints3.set_offsets(np.c_[x_[3],y_[3]])
            ipoints3.set_array(self.protein_intensities[i][3])
            ipoints4.set_offsets(np.c_[x_[4],y_[4]])
            ipoints4.set_array(self.protein_intensities[i][4])
            ipoints5.set_offsets(np.c_[x_[5],y_[5]])
            ipoints5.set_array(self.protein_intensities[i][5])
            ipoints6.set_offsets(np.c_[x_[6],y_[6]])
            ipoints6.set_array(self.protein_intensities[i][6])
            ipoints7.set_offsets(np.c_[x_[7],y_[7]])
            ipoints7.set_array(self.protein_intensities[i][7])
            ipoints8.set_offsets(np.c_[x_[8],y_[8]])
            ipoints8.set_array(self.protein_intensities[i][8])
            ipoints9.set_offsets(np.c_[x_[9],y_[9]])
            ipoints9.set_array(self.protein_intensities[i][9])
            ipoints10.set_offsets(np.c_[x_[10],y_[10]])
            ipoints10.set_array(self.protein_intensities[i][10])
            ipoints10_2.set_offsets(np.c_[(x_[10]+x_[11])/2, (y_[10]+y_[11])/2])
            ipoints10_2.set_array((self.protein_intensities[i][10]+self.protein_intensities[i][11])/2)
            ipoints11.set_offsets(np.c_[x_[11],y_[11]])
            ipoints11.set_array(self.protein_intensities[i][11])
            ipoints11_2.set_offsets(np.c_[(x_[11]+x_[12])/2, (y_[11]+y_[12])/2])
            ipoints11_2.set_array((self.protein_intensities[i][11]+self.protein_intensities[i][12])/2)
            ipoints12.set_offsets(np.c_[x_[12],y_[12]])
            ipoints12.set_array(self.protein_intensities[i][12])
            ipoints12_2.set_offsets(np.c_[(x_[12]+x_[13])/2, (y_[12]+y_[13])/2])
            ipoints12_2.set_array((self.protein_intensities[i][12]+self.protein_intensities[i][13])/2)
            ipoints13.set_offsets(np.c_[x_[13],y_[13]])
            ipoints13.set_array(self.protein_intensities[i][13])            
            ipoints13_2.set_offsets(np.c_[(x_[13]+x_[14])/2, (y_[13]+y_[14])/2])
            ipoints13_2.set_array((self.protein_intensities[i][13]+self.protein_intensities[i][14])/2)
            ipoints14.set_offsets(np.c_[x_[14],y_[14]])
            ipoints14.set_array(self.protein_intensities[i][14])
            ipoints14_2.set_offsets(np.c_[(x_[14]+x_[15])/2, (y_[14]+y_[15])/2])
            ipoints14_2.set_array((self.protein_intensities[i][14]+self.protein_intensities[i][15])/2)
            ipoints15.set_offsets(np.c_[x_[15],y_[15]])
            ipoints15.set_array(self.protein_intensities[i][15])
            ipoints15_2.set_offsets(np.c_[(x_[15]+x_[16])/2, (y_[15]+y_[16])/2])
            ipoints15_2.set_array((self.protein_intensities[i][15]+self.protein_intensities[i][16])/2)
            ipoints16.set_offsets(np.c_[x_[16],y_[16]])
            ipoints16.set_array(self.protein_intensities[i][16])
            ipoints16_2.set_offsets(np.c_[(x_[16]+x_[17])/2, (y_[16]+y_[17])/2])
            ipoints16_2.set_array((self.protein_intensities[i][16]+self.protein_intensities[i][17])/2)
            ipoints17.set_offsets(np.c_[x_[17],y_[17]])
            ipoints17.set_array(self.protein_intensities[i][17])
            ipoints17_2.set_offsets(np.c_[(x_[17]+x_[18])/2, (y_[17]+y_[18])/2])
            ipoints17_2.set_array((self.protein_intensities[i][17]+self.protein_intensities[i][18])/2)
            ipoints18.set_offsets(np.c_[x_[18],y_[18]])
            ipoints18.set_array(self.protein_intensities[i][18])
            ipoints18_2.set_offsets(np.c_[(x_[18]+x_[19])/2, (y_[18]+y_[19])/2])
            ipoints18_2.set_array((self.protein_intensities[i][18]+self.protein_intensities[i][19])/2)
            ipoints19.set_offsets(np.c_[x_[19],y_[19]])
            ipoints19.set_array(self.protein_intensities[i][19])
            ipoints19_2.set_offsets(np.c_[(x_[19]+x_[20])/2, (y_[19]+y_[20])/2])
            ipoints19_2.set_array((self.protein_intensities[i][19]+self.protein_intensities[i][20])/2)
            ipoints20.set_offsets(np.c_[x_[20],y_[20]])
            ipoints20.set_array(self.protein_intensities[i][20])
        
        
        if dark:
            plt.style.use('dark_background')
            plt.rcParams['savefig.facecolor'] = '#191919'
            plt.rcParams['figure.facecolor'] ='#191919'
            plt.rcParams['axes.facecolor'] = '#191919'
        else:
            plt.style.use('default')
        norm = plt.Normalize(vmin=0, vmax=1)
        fig, ax = plt.subplots()
        fig.suptitle(pc_name)
        (nu,) = plt.plot([], [], "b", lw=2, alpha=0.3)
        (cell,) = plt.plot([], [], "m", lw=2, alpha=0.3)
        if True:
            ipoints0 = plt.scatter([], [], c=[], norm=norm, s=point_size)
            ipoints1 = plt.scatter([], [], c=[], norm=norm, s=point_size)       
            ipoints2 = plt.scatter([], [], c=[], norm=norm, s=point_size)            
            ipoints3 = plt.scatter([], [], c=[], norm=norm, s=point_size)            
            ipoints4 = plt.scatter([], [], c=[], norm=norm, s=point_size)            
            ipoints5 = plt.scatter([], [], c=[], norm=norm, s=point_size)            
            ipoints6 = plt.scatter([], [], c=[], norm=norm, s=point_size)      
            ipoints7 = plt.scatter([], [], c=[], norm=norm, s=point_size)
            ipoints8 = plt.scatter([], [], c=[], norm=norm, s=point_size)       
            ipoints9 = plt.scatter([], [], c=[], norm=norm, s=point_size)            
            ipoints10 = plt.scatter([], [], c=[], norm=norm, s=point_size)           
            ipoints10_2 = plt.scatter([], [], c=[], norm=norm, s=point_size)
            ipoints11 = plt.scatter([], [], c=[], norm=norm, s=point_size)         
            ipoints11_2 = plt.scatter([], [], c=[], norm=norm, s=point_size)             
            ipoints12 = plt.scatter([], [], c=[], norm=norm, s=point_size)        
            ipoints12_2 = plt.scatter([], [], c=[], norm=norm, s=point_size)              
            ipoints13 = plt.scatter([], [], c=[], norm=norm, s=point_size)        
            ipoints13_2 = plt.scatter([], [], c=[], norm=norm, s=point_size)        
            ipoints14 = plt.scatter([], [], c=[], norm=norm, s=point_size)        
            ipoints14_2 = plt.scatter([], [], c=[], norm=norm, s=point_size)  
            ipoints15 = plt.scatter([], [], c=[], norm=norm, s=point_size)        
            ipoints15_2 = plt.scatter([], [], c=[], norm=norm, s=point_size)         
            ipoints16 = plt.scatter([], [], c=[], norm=norm, s=point_size)          
            ipoints16_2 = plt.scatter([], [], c=[], norm=norm, s=point_size)            
            ipoints17 = plt.scatter([], [], c=[], norm=norm, s=point_size)         
            ipoints17_2 = plt.scatter([], [], c=[], norm=norm, s=point_size)             
            ipoints18 = plt.scatter([], [], c=[], norm=norm, s=point_size)           
            ipoints18_2 = plt.scatter([], [], c=[], norm=norm, s=point_size)           
            ipoints19 = plt.scatter([], [], c=[], norm=norm, s=point_size)         
            ipoints19_2 = plt.scatter([], [], c=[], norm=norm, s=point_size)             
            ipoints20 = plt.scatter([], [], c=[], norm=norm, s=point_size)      
            
        ani = FuncAnimation(
            fig,
            update,
            self.stdpoints[pc_name] + self.stdpoints[pc_name][::-1],
            #fargs = (list(range(10))),
            init_func=init,
        )
        ax.axis("scaled")
        ax.set_facecolor('#541352FF')
        writer = PillowWriter(fps=3)
        ani.save(
            f"{save_dir}/{title}_{pc_name}.gif",
            writer=writer,
        )
        plt.close()

def display_scree_plot(pca, dark=True, save_dir="C:/Users/trang.le/Desktop/2D_shape_space/shapespace_plots"):
    """Display a scree plot for the pca"""

    scree = pca.explained_variance_ratio_ * 100
    plt.bar(np.arange(len(scree)) + 1, scree)
    plt.plot(np.arange(len(scree)) + 1, scree.cumsum(), c="red")

    if dark:
        plt.style.use('dark_background')
        plt.rcParams['savefig.facecolor'] = '#191919'
        plt.rcParams['figure.facecolor'] ='#191919'
        plt.rcParams['axes.facecolor'] = '#191919'
        #plt.set_facecolor('#191919')  
    for thres in [70, 80, 90, 95]:
        idx = np.searchsorted(scree.cumsum(), thres)
        plt.plot(idx + 1, scree.cumsum()[idx], c="red", marker="o")
        plt.annotate(f"{idx} PCs", xy=(idx + 3, scree.cumsum()[idx] - 5))
    plt.xlabel("Number of PCs")
    plt.ylabel("Percentage explained variance")
    plt.title("Scree plot")   
    plt.savefig(f"{save_dir}/PCA_scree.jpg", bbox_inches="tight")

    # plt.hlines(y=70, xmin = 0, xmax = len(scree), linestyles='dashed', alpha=0.5)
    # plt.vlines(x=np.argmax(scree.cumsum()>70), ymin = 0, ymax = 100, linestyles='dashed', alpha=0.5)
    # plt.hlines(y=80, xmin = 0, xmax = len(scree), linestyles='dashed', alpha=0.5)
    # plt.vlines(x=np.argmax(scree.cumsum()>80), ymin = 0, ymax = 100, linestyles='dashed', alpha=0.5)
    plt.show(block=False)
    plt.close()

def plot_interpolations(shape_path, pro_path,shift_dict, save_path, ori_fft, reduced_fft, n_coef, inverse_func):
    
    protein_ch = rotate(imread(pro_path), shift_dict["theta"])
    shapes = rotate(plt.imread(shape_path), shift_dict["theta"])
  
    fig, ax = plt.subplots(2, 3, figsize=(25,30))        
    ax[0,0].imshow(shapes)    
    ax[1,0].imshow(protein_ch)
    cell__ = []
    for fcoef in [ori_fft[: n_coef * 2], ori_fft[n_coef * 2 :]]: 
        ix__, iy__ = inverse_func(fcoef[:n_coef], fcoef[n_coef:])
        ax[0,0].scatter(iy__[0]+shift_dict["shift_c"][1],ix__[0]+shift_dict["shift_c"][0], color="w")
        ax[1,0].scatter(iy__[0]+shift_dict["shift_c"][1],ix__[0]+shift_dict["shift_c"][0], color="r")
        ax[0,1].scatter(ix__[0], iy__[0], color="r")
        ax[0,1].plot(ix__, iy__)
        ax[0,1].axis("scaled")
        cell__ += [np.concatenate([ix__, iy__])]

    x_,y_ = parameterize.get_coordinates(cell__[1].real, cell__[0].real, [0,0], n_isos = [3,7], plot=False)
    for (xi, yi) in zip(x_,y_):
        ax[0,2].plot(xi, yi, "--")
    ax[0,2].axis("scaled")
        
    fcoef_c = reduced_fft[0 : n_coef * 2]
    fcoef_n = reduced_fft[n_coef * 2 :]
    
    cell_ = []
    for fcoef in [fcoef_c, fcoef_n]:
        ix_, iy_ = inverse_func(fcoef[:n_coef], fcoef[n_coef:])
        ax[1,1].scatter(ix_[0], iy_[0], color="r")
        ax[1,1].plot(ix_, iy_)
        ax[1,1].axis("scaled")
        cell_ += [np.concatenate([ix_, iy_])]
    x_,y_ = parameterize.get_coordinates(cell_[1].real, cell_[0].real, [0,0], n_isos = [3,7], plot=False)
    for (xi, yi) in zip(x_,y_):
        ax[1,2].plot(xi, yi, "--")
    ax[1,2].axis("scaled")
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    
def plot_interpolation2(shape_path, pro_path,shift_dict, save_path, ori_fft, reduced_fft, n_coef, inverse_func):
    
    protein_ch = rotate(imread(pro_path), shift_dict["theta"])
    shapes = rotate(plt.imread(shape_path), shift_dict["theta"])
  
    fig, ax = plt.subplots(1, 3, figsize=(25,30))        
    ax[0].imshow(shapes)    
    ax[1].imshow(protein_ch)
    cell__ = []
    for fcoef in [ori_fft[: n_coef * 2], ori_fft[n_coef * 2 :]]: 
        ix__, iy__ = inverse_func(fcoef[:n_coef], fcoef[n_coef:])
        cell__ += [np.concatenate([ix__, iy__])]
    x_,y_ = parameterize.get_coordinates(cell__[1].real, cell__[0].real, [0,0], n_isos = [5,5], plot=False)
    for (xi, yi) in zip(x_,y_):
        ax[0].scatter(yi+shift_dict["shift_c"][1],xi+shift_dict["shift_c"][0], s=1, color="w")
        ax[1].scatter(yi+shift_dict["shift_c"][1],xi+shift_dict["shift_c"][0], s=1, color="w")
                    
    fcoef_c = reduced_fft[0 : n_coef * 2]
    fcoef_n = reduced_fft[n_coef * 2 :]
    
    cell_ = []
    for fcoef in [fcoef_c, fcoef_n]:
        ix_, iy_ = inverse_func(fcoef[:n_coef], fcoef[n_coef:])
        cell_ += [np.concatenate([ix_, iy_])]
    x_,y_ = parameterize.get_coordinates(cell_[1].real, cell_[0].real, [0,0], n_isos = [5,5], plot=False)
    for (xi, yi) in zip(x_,y_):
        ax[2].plot(xi, yi, "--")
    ax[2].axis("scaled")
    #plt.savefig(save_path, bbox_inches="tight")
    plt.close()

def plot_interpolation3(shape_path, pro_path,shift_dict, save_path, ori_fft, reduced_fft, n_coef, inverse_func):
    
    protein_ch = rotate(imread(pro_path), shift_dict["theta"])
    shapes = rotate(plt.imread(shape_path), shift_dict["theta"])
  
    fig, ax = plt.subplots(1, 4, figsize=(25,30))   
    fig.patch.set_facecolor('#191919')
    #fig.patch.set_alpha(1)
    ax[0].imshow(shapes, origin='lower') 
    #ax[0].set_facecolor('#191919')
    #ax[0].tight_axis()
    ax[1].imshow(protein_ch, origin='lower')
    ax[1].set_facecolor('#191919')
    cell__ = []
    for fcoef in [ori_fft[: n_coef * 2], ori_fft[n_coef * 2 :]]: 
        ix__, iy__ = inverse_func(fcoef[:n_coef], fcoef[n_coef:])
        cell__ += [np.concatenate([ix__, iy__])]
    x_,y_ = parameterize.get_coordinates(cell__[1].real, cell__[0].real, [0,0], n_isos = [10,10], plot=False)
    for i, (xi, yi) in enumerate(zip(x_,y_)):
        ax[0].scatter(yi+shift_dict["shift_c"][1],xi+shift_dict["shift_c"][0], s=0.5, alpha=0.3, color="w")
        ax[1].scatter(yi+shift_dict["shift_c"][1],xi+shift_dict["shift_c"][0], s=0.5, alpha=0.3, color="w")
        #ax[1].text(yi[i]+shift_dict["shift_c"][1], xi[i]+shift_dict["shift_c"][0], str(i))

    #Get intensity
    x = np.array(x_) + shift_dict["shift_c"][0]
    y = np.array(y_) + shift_dict["shift_c"][1]
    m = parameterize.get_intensity(protein_ch, x, y, k=7)
    m_normed = m/m.max()
    norm = plt.Normalize(vmin=0, vmax=1)
    print(m_normed.min(),m_normed.max(), m_normed[1,:])
    for i, (xi, yi) in enumerate(zip(x_,y_)):
        ax[2].scatter(yi+shift_dict["shift_c"][1], xi+shift_dict["shift_c"][0], c=m_normed[i,:], norm=norm,s=2)
        #ax[2].text(yi[i]+shift_dict["shift_c"][1], xi[i]+shift_dict["shift_c"][0], str(i))
    ax[2].axis("scaled")
    #ax[2].set_facecolor("#541352FF")   
    ax[2].set_facecolor("#191919")   
    
    fcoef_c = reduced_fft[0 : n_coef * 2]
    fcoef_n = reduced_fft[n_coef * 2 :]
    cell_ = []
    for fcoef in [fcoef_c, fcoef_n]:
        ix_, iy_ = inverse_func(fcoef[:n_coef], fcoef[n_coef:])
        cell_ += [np.concatenate([ix_, iy_])]
    x_,y_ = parameterize.get_coordinates(cell_[1].real, cell_[0].real, [0,0], n_isos = [10,10], plot=False)
    for i, (xi, yi) in enumerate(zip(x_,y_)):
        ax[3].plot(xi, yi, "--", alpha=0.3)
        ax[3].scatter(xi, yi,c=m_normed[i,:],norm=norm)
    ax[3].axis("scaled")
    ax[3].set_facecolor('#191919')
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()

def get_protein_intensity(pro_path, shift_dict, ori_fft, n_coef, inverse_func, fourier_algo = "fft"):
    
    protein_ch = rotate(imread(pro_path), shift_dict["theta"])
    #shapes = rotate(plt.imread(shape_path), shift_dict["theta"])
    #protein_ch = exposure.equalize_hist(protein_ch)
    #thresh = threshold_mean(protein_ch)
    #protein_ch[protein_ch < thresh] = 0 
    
    cell__ = []
    if fourier_algo =="fft":
        n_coef_= n_coef*2
    elif fourier_algo == "efd":
        n_coef_= n_coef*4 + 2
    for fcoef in [ori_fft[: n_coef_*2], ori_fft[n_coef_*2 :]]: 
        ix__, iy__ = inverse_func(fcoef[:n_coef_], fcoef[n_coef_:])
        cell__ += [np.concatenate([ix__, iy__])]
    x_,y_ = parameterize.get_coordinates(cell__[1].real, cell__[0].real, [0,0], n_isos = [10,20], plot=False)

    #Get intensity
    x = np.array(x_) + shift_dict["shift_c"][0]
    y = np.array(y_) + shift_dict["shift_c"][1]
    m = parameterize.get_intensity(protein_ch, x, y, k=5)
    m_normed = m#/m.max()
    return m_normed
    

def _plot_protein_through_shape_variation_gif(pc_name, nu_coords, mem_coords, protein_intensities, title='', dark=True, point_size=4,save_dir="C:/Users/trang.le/Desktop/2D_shape_space/shapespace_plots"):
    def init():
        """Local function to init space in animated plots"""
        ax.set_xlim(-600, 600)
        ax.set_ylim(-650, 600)

    def update(i):
        x_,y_ = parameterize.get_coordinates(nu_coords[i], mem_coords[i], [0,0], n_isos = [10,10], plot=False)
        nu.set_data(x_[10],y_[10])
        cell.set_data(x_[-1],y_[-1])
        ipoints0.set_offsets(np.c_[x_[0],y_[0]])
        ipoints0.set_array(protein_intensities[i][0])
        ipoints1.set_offsets(np.c_[x_[1],y_[1]])
        ipoints1.set_array(protein_intensities[i][1])
        ipoints2.set_offsets(np.c_[x_[2],y_[2]])
        ipoints2.set_array(protein_intensities[i][2])
        ipoints3.set_offsets(np.c_[x_[3],y_[3]])
        ipoints3.set_array(protein_intensities[i][3])
        ipoints4.set_offsets(np.c_[x_[4],y_[4]])
        ipoints4.set_array(protein_intensities[i][4])
        ipoints5.set_offsets(np.c_[x_[5],y_[5]])
        ipoints5.set_array(protein_intensities[i][5])
        ipoints6.set_offsets(np.c_[x_[6],y_[6]])
        ipoints6.set_array(protein_intensities[i][6])
        ipoints7.set_offsets(np.c_[x_[7],y_[7]])
        ipoints7.set_array(protein_intensities[i][7])
        ipoints8.set_offsets(np.c_[x_[8],y_[8]])
        ipoints8.set_array(protein_intensities[i][8])
        ipoints9.set_offsets(np.c_[x_[9],y_[9]])
        ipoints9.set_array(protein_intensities[i][9])
        ipoints10.set_offsets(np.c_[x_[10],y_[10]])
        ipoints10.set_array(protein_intensities[i][10])
        ipoints10_2.set_offsets(np.c_[(x_[10]+x_[11])/2, (y_[10]+y_[11])/2])
        ipoints10_2.set_array((protein_intensities[i][10]+protein_intensities[i][11])/2)
        ipoints11.set_offsets(np.c_[x_[11],y_[11]])
        ipoints11.set_array(protein_intensities[i][11])
        ipoints11_2.set_offsets(np.c_[(x_[11]+x_[12])/2, (y_[11]+y_[12])/2])
        ipoints11_2.set_array((protein_intensities[i][11]+protein_intensities[i][12])/2)
        ipoints12.set_offsets(np.c_[x_[12],y_[12]])
        ipoints12.set_array(protein_intensities[i][12])
        ipoints12_2.set_offsets(np.c_[(x_[12]+x_[13])/2, (y_[12]+y_[13])/2])
        ipoints12_2.set_array((protein_intensities[i][12]+protein_intensities[i][13])/2)
        ipoints13.set_offsets(np.c_[x_[13],y_[13]])
        ipoints13.set_array(protein_intensities[i][13])            
        ipoints13_2.set_offsets(np.c_[(x_[13]+x_[14])/2, (y_[13]+y_[14])/2])
        ipoints13_2.set_array((protein_intensities[i][13]+protein_intensities[i][14])/2)
        ipoints14.set_offsets(np.c_[x_[14],y_[14]])
        ipoints14.set_array(protein_intensities[i][14])
        ipoints14_2.set_offsets(np.c_[(x_[14]+x_[15])/2, (y_[14]+y_[15])/2])
        ipoints14_2.set_array((protein_intensities[i][14]+protein_intensities[i][15])/2)
        ipoints15.set_offsets(np.c_[x_[15],y_[15]])
        ipoints15.set_array(protein_intensities[i][15])
        ipoints15_2.set_offsets(np.c_[(x_[15]+x_[16])/2, (y_[15]+y_[16])/2])
        ipoints15_2.set_array((protein_intensities[i][15]+protein_intensities[i][16])/2)
        ipoints16.set_offsets(np.c_[x_[16],y_[16]])
        ipoints16.set_array(protein_intensities[i][16])
        ipoints16_2.set_offsets(np.c_[(x_[16]+x_[17])/2, (y_[16]+y_[17])/2])
        ipoints16_2.set_array((protein_intensities[i][16]+protein_intensities[i][17])/2)
        ipoints17.set_offsets(np.c_[x_[17],y_[17]])
        ipoints17.set_array(protein_intensities[i][17])
        ipoints17_2.set_offsets(np.c_[(x_[17]+x_[18])/2, (y_[17]+y_[18])/2])
        ipoints17_2.set_array((protein_intensities[i][17]+protein_intensities[i][18])/2)
        ipoints18.set_offsets(np.c_[x_[18],y_[18]])
        ipoints18.set_array(protein_intensities[i][18])
        ipoints18_2.set_offsets(np.c_[(x_[18]+x_[19])/2, (y_[18]+y_[19])/2])
        ipoints18_2.set_array((protein_intensities[i][18]+protein_intensities[i][19])/2)
        ipoints19.set_offsets(np.c_[x_[19],y_[19]])
        ipoints19.set_array(protein_intensities[i][19])
        ipoints19_2.set_offsets(np.c_[(x_[19]+x_[20])/2, (y_[19]+y_[20])/2])
        ipoints19_2.set_array((protein_intensities[i][19]+protein_intensities[i][20])/2)
        ipoints20.set_offsets(np.c_[x_[20],y_[20]])
        ipoints20.set_array(protein_intensities[i][20])
    
    
    if dark:
        plt.style.use('dark_background')
        plt.rcParams['savefig.facecolor'] = '#191919'
        plt.rcParams['figure.facecolor'] ='#191919'
        plt.rcParams['axes.facecolor'] = '#191919'
    else:
        plt.style.use('default')
    norm = plt.Normalize(vmin=0, vmax=1)
    fig, ax = plt.subplots()
    fig.suptitle(pc_name)
    (nu,) = plt.plot([], [], "b", lw=2, alpha=0.3)
    (cell,) = plt.plot([], [], "m", lw=2, alpha=0.3)
    if True:
        ipoints0 = plt.scatter([], [], c=[], norm=norm, s=point_size)
        ipoints1 = plt.scatter([], [], c=[], norm=norm, s=point_size)       
        ipoints2 = plt.scatter([], [], c=[], norm=norm, s=point_size)            
        ipoints3 = plt.scatter([], [], c=[], norm=norm, s=point_size)            
        ipoints4 = plt.scatter([], [], c=[], norm=norm, s=point_size)            
        ipoints5 = plt.scatter([], [], c=[], norm=norm, s=point_size)            
        ipoints6 = plt.scatter([], [], c=[], norm=norm, s=point_size)      
        ipoints7 = plt.scatter([], [], c=[], norm=norm, s=point_size)
        ipoints8 = plt.scatter([], [], c=[], norm=norm, s=point_size)       
        ipoints9 = plt.scatter([], [], c=[], norm=norm, s=point_size)            
        ipoints10 = plt.scatter([], [], c=[], norm=norm, s=point_size)           
        ipoints10_2 = plt.scatter([], [], c=[], norm=norm, s=point_size)
        ipoints11 = plt.scatter([], [], c=[], norm=norm, s=point_size)         
        ipoints11_2 = plt.scatter([], [], c=[], norm=norm, s=point_size)             
        ipoints12 = plt.scatter([], [], c=[], norm=norm, s=point_size)        
        ipoints12_2 = plt.scatter([], [], c=[], norm=norm, s=point_size)              
        ipoints13 = plt.scatter([], [], c=[], norm=norm, s=point_size)        
        ipoints13_2 = plt.scatter([], [], c=[], norm=norm, s=point_size)        
        ipoints14 = plt.scatter([], [], c=[], norm=norm, s=point_size)        
        ipoints14_2 = plt.scatter([], [], c=[], norm=norm, s=point_size)  
        ipoints15 = plt.scatter([], [], c=[], norm=norm, s=point_size)        
        ipoints15_2 = plt.scatter([], [], c=[], norm=norm, s=point_size)         
        ipoints16 = plt.scatter([], [], c=[], norm=norm, s=point_size)          
        ipoints16_2 = plt.scatter([], [], c=[], norm=norm, s=point_size)            
        ipoints17 = plt.scatter([], [], c=[], norm=norm, s=point_size)         
        ipoints17_2 = plt.scatter([], [], c=[], norm=norm, s=point_size)             
        ipoints18 = plt.scatter([], [], c=[], norm=norm, s=point_size)           
        ipoints18_2 = plt.scatter([], [], c=[], norm=norm, s=point_size)           
        ipoints19 = plt.scatter([], [], c=[], norm=norm, s=point_size)         
        ipoints19_2 = plt.scatter([], [], c=[], norm=norm, s=point_size)             
        ipoints20 = plt.scatter([], [], c=[], norm=norm, s=point_size)      
    
    n_bins = protein_intensities.shape[0]
    steps = list(range(n_bins))
    ani = FuncAnimation(
        fig,
        update,
        steps + steps[::-1],
        init_func=init,
    )
    ax.axis("scaled")
    ax.set_facecolor('#541352FF')
    writer = PillowWriter(fps=3)
    ani.save(
        f"{save_dir}/{title}_{pc_name}.gif",
        writer=writer,
    )
    plt.close()


def plot_example_cells(bin_links, n_coef=128, cells_per_bin=5, shape_coef_path="", save_path=None):
    plt.figure()
    fig, ax = plt.subplots(cells_per_bin, len(bin_links),sharex=True,sharey=True) # (number of random cells, number of  bin)
    for b_index, b_ in enumerate(bin_links):
        cells_ = np.random.choice(b_, cells_per_bin)
        for i, c in enumerate(cells_):
            fft_coefs = get_line(shape_coef_path, search_text=c, mode="first")
            f_coef_n = fft_coefs.split(",")[1:n_coef*2+1]
            f_coef_n = [complex(s.replace('i', 'j')) for s in f_coef_n]
            f_coef_c = fft_coefs.split(",")[n_coef*2+1:]
            f_coef_c = [complex(s.replace('i', 'j')) for s in f_coef_c]
            ix_n, iy_n = coefs.inverse_fft(f_coef_n[0:n_coef],f_coef_n[n_coef:])
            ix_c, iy_c = coefs.inverse_fft(f_coef_c[0:n_coef],f_coef_c[n_coef:])
            ax[i, b_index].plot(ix_n.real, iy_n.real)
            ax[i, b_index].plot(ix_c.real, iy_c.real)
            ax[i, b_index].axis("scaled")
    if save_path != None:
        fig.savefig(save_path, bbox_inches=None)
        plt.close()