import configs.config_callisto as cfg

# Define the target rule that executes the entire pipeline
rule all:
    input:
        f"{cfg.PROJECT_DIR}/shapemode/{cfg.ALIGNMENT}_{cfg.MODE}/cells_assigned_to_pc_bins.json"

rule coefficient:
    input:
        script = "coefficients/s2_calculate_fft.py"
    output:
        f"{cfg.PROJECT_DIR}/fftcoefs/{cfg.ALIGNMENT}/fftcoefs_128.txt",
        f"{cfg.PROJECT_DIR}/fftcoefs/{cfg.ALIGNMENT}/shift_error_meta_fft128.txt"
    shell:
        """
        cd coefficients
        python s2_calculate_fft.py
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
        python s3_calculate_shapemodes.py
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
        python cell_nucleus_ratio.py
        cd ..
        """
