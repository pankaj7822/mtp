import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import shutil
import sys
import pickle
from sklearn.decomposition import NMF
from MEMD_all import memd
import concurrent.futures

def generateimf(file_path,save_path):
    print("started working for file",file_path)
    df=pd.read_csv(file_path,header=None)
    n_array=df.to_numpy()
    imf = memd(n_array)
    if not os.path.exists(save_path):
        pickle_out = open(save_path,"wb")
        pickle.dump(imf, pickle_out)
    print("Task Completed for",save_path)

if __name__ == "__main__":
    if sys.argv[1]=="gi":
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            for i in range(1,10):
                for j in ["E","T"]:
                    for k in range(1,5):
                        dir_path=os.path.join("segmented_data",str(i),str(k),j,"A0"+str(i)+j+"_"+str(k))
                        segment_files=os.listdir(dir_path)
                        for f in segment_files:
                            executor.submit(generateimf,os.path.join(dir_path,f),os.path.join("imfs",f[:-4]+"p"+".pickle"))