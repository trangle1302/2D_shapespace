import argparse 
import sys
sys.path.append("..")
import configs.config_sherlock as cfg

def generate_sh():
    # remove instructions from previous run
    try:
        os.remove("run.sh")
        print(f"run.sh has been removed successfully.")
    except OSError as error:
        print(f"Error: run.sh - {error.strerror}")
    
    # create new instruction files
    with open("submit_s4_protein.sh") as fs:
        for pc_ in range(1,7):
            for b_ in range(7):
                fs.write(f"srun -N 1 -n1 python3 avg_protein.py --merged_bins {b_} --pc PC{pc_} &\n")

if __name__ == "__main__":
    generate_sh()
