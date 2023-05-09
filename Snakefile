import config.config_callisto as cfg

rule coefficient:
    input:
        script = "coefficients/s2_calculate_fft.py"
        data_dir = f"{cfg.PROJECT_DIR}/cell_masks/*.npy"
    output:
        f"{cfg.PROJECT_DIR}/coefficients/fftcoefs/{cfg.ALIGNMENT}/fftcoefs_128.txt",
        f"{cfg.PROJECT_DIR}/coefficients/fftcoefs/{cfg.ALIGNMENT}/shift_error_meta_fft128.txt"
    shell:
        """
        cd coefficients
        python s2_calculate_fft.py
        cd ..
        """

rule shapemode:
    input:
        f"{cfg.PROJECT_DIR}/coefficients/fftcoefs/{cfg.ALIGNMENT}/fftcoefs_128.txt",
        f"{cfg.PROJECT_DIR}/coefficients/fftcoefs/{cfg.ALIGNMENT}/shift_error_meta_fft128.txt"
    output:
        f"{cfg.PROJECT_DIR}/coefficients/fftcoefs/{cfg.ALIGNMENT}_{cfg.MODE}/cells_assigned_to_pc_bins.json"
    shell:
        """
        cd shapemodes
        python s3_calculate_shapemodes.py
        cd ..
        """

rule cell_nu_ratio:
    input:
        f"{cfg.PROJECT_DIR}/coefficients/fftcoefs/{cfg.ALIGNMENT}/fftcoefs_128.txt",
        f"{cfg.PROJECT_DIR}/coefficients/fftcoefs/{cfg.ALIGNMENT}/shift_error_meta_fft128.txt"
    output:
        f"{cfg.PROJECT_DIR}/cell_nu_ratio.txt"
    shell:
        """
        cd shapemodes
        python s3_calculate_shapemodes.py
        cd ..
        """

# Define the target rule that executes the entire pipeline
rule all:
    input:
        f"{cfg.PROJECT_DIR}/coefficients/fftcoefs/{cfg.ALIGNMENT}_{cfg.MODE}/cells_assigned_to_pc_bins.json"
