import sys
sys.path.append("..")
import numpy as np
import configs.config_all as cfg
import matplotlib.pyplot as plt
from utils.helpers import equidistance

def main():
    fig, ax = plt.subplots(1, len(cfg.CELL_LINE), sharex=True, sharey=True)
    plt.subplots_adjust(wspace=0, hspace=0)    
    for j, cell_line in enumerate(cfg.CELL_LINE):        
        shapemode_dir = f"{cfg.PROJECT_DIR}/{cell_line}/shapemode/{cfg.ALIGNMENT}_{cfg.MODE}"

        avg_coords = np.load(f"{shapemode_dir}/Avg_cell.npz")                
        ix_n = avg_coords["ix_n"]
        iy_n = avg_coords["iy_n"]
        ix_c = avg_coords["ix_c"]
        iy_c = avg_coords["iy_c"]
        min_x = ix_c.min()
        min_y = iy_c.min()
        ix_n -= min_x
        iy_n -= min_y
        ix_c -= min_x
        iy_c -= min_y
        ix_n, iy_n = equidistance(ix_n.real, iy_n.real, cfg.N_COEFS * 10)
        ix_c, iy_c = equidistance(ix_c.real, iy_c.real, cfg.N_COEFS * 10)
        ax[j].plot(ix_n, iy_n, "#8ab0cf")
        ax[j].plot(ix_c, iy_c, "m")
        ax[j].set_title(cell_line, size=10)
    fig.savefig(f"{cfg.PROJECT_DIR}/Average_cell_panel.png")

if __name__ == "__main__":
    main()