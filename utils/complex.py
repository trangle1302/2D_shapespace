import os
import random
import numpy as np
import pandas as pd


def read_complex_df(fft_dir="", n_coef=128, n_samples = 10000):
    """ Function to read dataframe of complex numbers (fft coefficients) from .txt file
    Args:
        fft_dir: path to .txt file
        n_coef: number of columns x 2
        n_cv: number of cross validation
        n_samples: number of random samples to load for 1 CV
    
    Returns:
        df: dataframe of coefficient, sample name as row indexes
    """
    fft_path = os.path.join(fft_dir,f"fftcoefs_{n_coef}.txt")
    with open(fft_path) as f:
        count = sum(1 for _ in f)
    #for i in range(n_cv):
    with open(fft_path, "r") as file:
        lines = dict()
        if n_samples ==-1:
            specified_lines = range(count)
        else:
            specified_lines = random.sample(range(count), n_samples) # 10k cells/ CV
        # loop over lines in a file
        for pos, l_num in enumerate(file):
            # check if the line number is specified in the lines to read array
            #print(pos)
            if pos in specified_lines:
                # print the required line number
                data_ = l_num.strip().split(',')
                if len(data_[1:]) != n_coef*4:
                    continue
                #data_dict = {data_dict[0]:data_dict[1:]}
                lines[data_[0]]=data_[1:]

    df = pd.DataFrame(lines).transpose()
    df = df.applymap(lambda s: complex(s.replace('i', 'j'))) 

    return df
    