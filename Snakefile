import configs.config_sherlock as cfg
import glob

# Define the target rule that executes the entire pipeline
rule all:
    input:
        f"{cfg.PROJECT_DIR}/shapemode/{cfg.ALIGNMENT}_{cfg.MODE}/cells_assigned_to_pc_bins.json",
        #f"{cfg.PROJECT_DIR}/morphed_protein_avg/PC1/Microtubules_bin6.png"

rule coefficient:
    input:
        script = "coefficients/s2_calculate_fft.py"
    output:
        f"{cfg.PROJECT_DIR}/fftcoefs/{cfg.ALIGNMENT}/fftcoefs_128.txt",
        f"{cfg.PROJECT_DIR}/fftcoefs/{cfg.ALIGNMENT}/shift_error_meta_fft128.txt"
    shell:
        """
        cd coefficients
        sbatch submit_s2.sh
        cd ..
        """

rule shapemode:
    input:
        f"{cfg.PROJECT_DIR}/fftcoefs/{cfg.ALIGNMENT}/fftcoefs_128.txt",
        f"{cfg.PROJECT_DIR}/fftcoefs/{cfg.ALIGNMENT}/shift_error_meta_fft128.txt",
        f"{cfg.PROJECT_DIR}/cell_nu_ratio.txt",
        f"{cfg.META_PATH}"
    output:
        f"{cfg.PROJECT_DIR}/shapemode/{cfg.ALIGNMENT}_{cfg.MODE}/cells_assigned_to_pc_bins.json"
    shell:
        """
        cd shapemodes
        sbatch submit_s3.sh
        cd ..
        """

rule cell_nu_ratio:
    input:
        f"{cfg.PROJECT_DIR}/fftcoefs/{cfg.ALIGNMENT}/fftcoefs_128.txt",
        f"{cfg.PROJECT_DIR}/fftcoefs/{cfg.ALIGNMENT}/shift_error_meta_fft128.txt"
    output:
        f"{cfg.PROJECT_DIR}/cell_nu_ratio.txt"
    shell:
        """
        cd analysis
        module load python/3.9.0 
        module load py-pandas/2.0.1_py39
        pip install joblib
        srun --nodes=1 --ntasks=1 --mem=1G --time=01:00:00 python3 cell_nucleus_ratio.py
        cd ..
        """

rule organelle:
    input:
        f"{cfg.PROJECT_DIR}/shapemode/{cfg.ALIGNMENT}_{cfg.MODE}/cells_assigned_to_pc_bins.json"
    output:
        [f"{cfg.PROJECT_DIR}/morphed_protein_avg/PC{pc_}/{org}_bin{b}.png" for b in range(6) for org in cfg.ORGANELLES for pc_ in range(1,7)]
    shell:
        """
        cd warps
        module load python/3.9.0
        srun --nodes=1 --ntasks=1 --mem=5G --time=01:00:00 python3 generate_runs.py
        bash run.sh
        cd ..
        """
