import os
import sys
sys.path.append("..")
import configs.config_callisto as cfg

def generate_script():
    # remove instructions from previous run
    try:
        os.remove("run.sh")
        print(f"run.sh has been removed successfully.")
    except OSError as error:
        print(f"Error: run.sh - {error.strerror}")

    # create a new instruction file
    with open('run.sh', 'a') as fs:
        for org in cfg.ORGANELLES:
            for pc_ in range(1,7):
                all_present = all([os.path.exists(f"{cfg.PROJECT_DIR}/morphed_protein_avg/PC{pc_}/{org}_bin{b}.png") for b in range(7)])
                if not all_present:
                    fs.write(f"python3 avg_organelle.py --org {org}  --pc PC{pc_}\n")
    print(f"New run.sh has been created successfully.")

if __name__ == "__main__":
    generate_script()