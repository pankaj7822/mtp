import sys
import pickle
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import os
import numpy


val=""
files=os.listdir("imfs")
E_df=pd.DataFrame()
T_df=pd.DataFrame()
for f in files:
    subject=int(f[2])
    c_type=f[3]
    c_class=f[5]
    p = open(os.path.join("imfs",f),'rb')
    imf = pickle.load(p)
    if(int(imf.shape[0])<9):
        val+=f+" Imfs: "+str(imf.shape[0])+"\n"
        pass
    x=imf[:9,0,:]
    print(imf.shape,f)
    scaler = StandardScaler()
    scaler.fit(x)
    x = scaler.transform(x)
    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(x)
    principalDf = pd.DataFrame(data = principalComponents)
    f_narray=principalDf.to_numpy()
    for i in range(1,22):
        x=imf[:9,i,:]
        scaler = StandardScaler()
        scaler.fit(x)
        x = scaler.transform(x)
        pca = PCA(n_components=2)
        principalComponents = pca.fit_transform(x)
        principalDf = pd.DataFrame(data = principalComponents)
        narray=principalDf.to_numpy()
        f_narray=np.hstack((f_narray,narray))
        # print(pca.explained_variance_ratio_,sum(pca.explained_variance_ratio_))

    f_narray=f_narray.flatten()
    f_narray=numpy.append(f_narray,subject)
    f_narray=numpy.append(f_narray,c_class)
    print(f_narray.shape,f)
    df=pd.DataFrame(f_narray.reshape(-1, len(f_narray)))
    if c_type=="T":
        T_df=T_df.append(df)
    elif c_type=="E":
        E_df=E_df.append(df)
print(T_df.shape)
print(E_df.shape)
T_df.to_csv("T_IMF.csv",header=None,index=False)
E_df.to_csv("E_IMF.csv",header=None,index=False)
f = open("excluded_segments.txt", "w")
f.write(val)
f.close()