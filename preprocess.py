import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
import os
import sys
import concurrent.futures
from sklearn.decomposition import FastICA
import pickle

raw_base_path="raw_data"
reduced_base_path="reduced_data"
visualization_base_path="visualizations"
files=sorted(list(os.listdir("reduced_data")))
def getmetadata(f):
    temp=list(f.split("."))
    name=temp[0]
    subject=name[2]
    mov_class=name[-1]
    ds_type=name[3]
    meta_data={"subject":subject,"mov_class":mov_class,"ds_type":ds_type}
    return meta_data

def createreduced(raw_base_path,reduced_base_path,f):
    raw_path=os.path.join(raw_base_path,f)
    reduced_path=os.path.join(reduced_base_path,f)
    df=pd.read_csv(raw_path,header=None)
    df=df.iloc[:,:-3]
    print("Working on ",f)
    if not os.path.exists(reduced_base_path):
        os.mkdir(reduced_base_path)
    df.to_csv(reduced_path,header=None,index=False)

def refine_data_butterworth(reduced_base_path,save_path,f,fs=250,fc=100):
    print("Butterworth refining started for ",f)
    file_path=os.path.join(reduced_base_path,f)
    save_path=os.path.join(save_path,f)
    df=pd.read_csv(file_path,header=None)
    df2=pd.DataFrame()
    for i in range(len(df.columns)):
        data=list(df[:][i]) # Generate data properly  
        w = fc / (fs / 2)  # Normalize the frequency
        b, a = signal.butter(5, w, 'low') #5th order low banpass filter
        butterworth_output = signal.filtfilt(b, a, data)
        df2[i]=butterworth_output
    df2.to_csv(save_path,index=False,header=None)
    print("Butterworth Refined Data for ",f)    
        
def refine_data_golay(reduced_base_path,save_path,f,window_size=7,poly_order=3):
    print("Golay refining started for ",f)
    file_path=os.path.join(reduced_base_path,f)
    save_path=os.path.join(save_path,f)
    df=pd.read_csv(file_path,header=None)
    df2=pd.DataFrame()
    for i in range(len(df.columns)):
        data=list(df[:][i]) # Generate data properly  
        golay_output= signal.savgol_filter(data, window_size, poly_order)
        df2[i]=golay_output
    df2.to_csv(save_path,index=False,header=None)
    print("Golay Refined Data for ",f)    

def plotgraphs(file_path,save_path):
    df=pd.read_csv(file_path,header=None)
    save_path=os.path.join(save_path,'voltage_timestamp.tiff')
    print("File Path: ",file_path,"Save Path : ",save_path)
    for j in range(len(df.columns)):
        l=list(df[:][j])
        n=[]
        temp=0
        for i in range(len(l)):
            n.append(temp)
            temp+=15
        plt.plot(n,l,label=j+1)
        plt.xlabel("Time Stamps")
        plt.ylabel("Voltage")
        plt.title('Voltage Vs Time Stamps',loc='left')
        plt.legend(loc='upper right',columnspacing=0.3,bbox_to_anchor=(1.10,1.15),
            ncol=11, fancybox=True, labelspacing=0.3,handlelength=0.4,handletextpad=0.1)
    plt.savefig(save_path,format="tiff",dpi=350)
    plt.close()
    print("Plotted for ",save_path)

def removeartifacts(file_path,save_path=""):
    print("removing artifacts for",file_path)
    ica = FastICA(n_components=22)
    df=pd.read_csv(file_path,header=None)
    comps = pd.DataFrame(ica.fit_transform(df))
    cm=pd.DataFrame(comps)
    cm.to_csv(save_path,index=False,header=None)
    print("artifacts removed for",save_path)
    # time_stamps=[i*15 for i in range(len(list(cm[:][0])))]
    # print(time_stamps)
    # plt.plot(time_stamps,list(df[:][0]),label="golay")
    # plt.xlabel("Time Stamps")
    # plt.ylabel("Voltage")
    # plt.title('Voltage Vs Time Stamps',loc='left')
    # plt.savefig("golayplot.tiff",format="tiff",dpi=350)
    # plt.close()
    # plt.plot(time_stamps,list(cm[:][0]),label="artifacts_removed")
    # plt.xlabel("Time Stamps")
    # plt.ylabel("Voltage")
    # plt.title('Voltage Vs Time Stamps',loc='left')
    # plt.savefig("artifactsremoved.tiff",format="tiff",dpi=350)
    # plt.close()

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = signal.lfilter(b, a, data)
    return y

    
def centralbetafrequency(file_path,save_path):
    print("working on ",file_path)
    df=pd.read_csv(file_path,header=None)
    df2=pd.DataFrame()
    for i in range(len(df.columns)):
        y=butter_bandpass_filter(df[:][i],12,30,250)
        df2[i]=y
    df2.to_csv(save_path,header=None,index=False)    

def segment_data(file_path):
    print("Segmenting ",file_path)
    temp=file_path.split("/")
    df=pd.read_csv(f,header=None)
    start=0
    end=min_n
    count=0
    for i in range(int(min_n/1000)):
        count+=1
        j=start
        k=min(j+1000,end)
        temp_df=df[j:k]
        temp_df.to_csv(os.path.join("segmented_data",temp[1],temp[2],temp[3],temp[4][:-4],temp[4][:-4]+"part_"+str(count)+".csv"),header=None,index=False)
        start+=1000
    print("Segmented ",file_path)
 
refined_golay_filepaths=[]
artifacts_removed_filepaths=[]
beta_extracted_filepaths=[]
base_golay_path=os.path.join("refined_data","golay")
base_artifacts_removed_path="artifacts_removed_data"
base_beta_extracted_path="beta_extracted"
base_channel_selected_path="channel_selected"
channel_selected_file_paths=[]
for i in range(1,10):
    for j in range(1,5):
        for t in ["E","T"]:
            file_name="A0"+str(i)+t+"_"+str(j)+".csv"
            
            p=os.path.join(base_golay_path,str(i),str(j),str(t),file_name)
            refined_golay_filepaths.append(p)
            
            q=os.path.join(base_artifacts_removed_path,str(i),str(j),str(t),file_name)
            artifacts_removed_filepaths.append(q)

            r=os.path.join(base_beta_extracted_path,str(i),str(j),str(t),file_name)
            beta_extracted_filepaths.append(r)

            s=os.path.join(base_channel_selected_path,str(i),str(j),str(t),file_name)
            channel_selected_file_paths.append(r)
            
if __name__ == "__main__":
    if sys.argv[1]=="r":
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            for f in files:
                try:
                    executor.submit(createreduced,raw_base_path,reduced_base_path,f)
                except Exception as e:
                    print("Exception: ",e)
    if sys.argv[1]=="rv":
        raw_visualization_path=os.path.join(visualization_base_path,"raw_visualization")
        if not os.path.exists(visualization_base_path):
            os.mkdir(visualization_base_path)
        if not os.path.exists(raw_visualization_path):
            os.mkdir(raw_visualization_path)
        for f in files:
            meta_data=getmetadata(f)
            subject_path=os.path.join(raw_visualization_path,meta_data["subject"])
            class_path=os.path.join(subject_path,meta_data["mov_class"])
            type_path=os.path.join(class_path,meta_data["ds_type"])
            if not os.path.exists(subject_path):
                os.mkdir(subject_path)
            if not os.path.exists(class_path):
                os.mkdir(class_path)
            if not os.path.exists(type_path):
                os.mkdir(type_path)
            try:
                file_path=os.path.join(reduced_base_path,f)
                save_path=type_path
                plotgraphs(file_path,save_path)
            except Exception as e:
                print("Exception: ",e)
    if sys.argv[1]=="rf":
        if not os.path.exists("refined_data"):
            os.mkdir("refined_data")
        butterworth_path=os.path.join("refined_data","butter_worth")
        golay_path=os.path.join("refined_data","golay")
        if not os.path.exists(butterworth_path):
            os.mkdir(butterworth_path)
        if not os.path.exists(golay_path):
            os.mkdir(golay_path)
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            for f in files:
                meta_data=getmetadata(f)
                subject_path=os.path.join(butterworth_path,meta_data["subject"])
                class_path=os.path.join(subject_path,meta_data["mov_class"])
                type_path=os.path.join(class_path,meta_data["ds_type"])
                subject_path2=os.path.join(golay_path,meta_data["subject"])
                class_path2=os.path.join(subject_path2,meta_data["mov_class"])
                type_path2=os.path.join(class_path2,meta_data["ds_type"])
                if not os.path.exists(subject_path):
                    os.mkdir(subject_path)
                if not os.path.exists(class_path):
                    os.mkdir(class_path)
                if not os.path.exists(type_path):
                    os.mkdir(type_path)
                if not os.path.exists(subject_path2):
                    os.mkdir(subject_path2)
                if not os.path.exists(class_path2):
                    os.mkdir(class_path2)
                if not os.path.exists(type_path2):
                    os.mkdir(type_path2)
                try:
                    save_path1=type_path
                    save_path2=type_path2
                    executor.submit(refine_data_butterworth,reduced_base_path,save_path1,f)
                    executor.submit(refine_data_golay,reduced_base_path,save_path2,f)
                except Exception as e:
                    print("Exception: ",e)
    if sys.argv[1]=="ar":
        if not os.path.exists("artifacts_removed_data"):
            os.mkdir("artifacts_removed_data")
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            for f in refined_golay_filepaths:
                temp=f.split("/")
                save_path=os.path.join("artifacts_removed_data",temp[2],temp[3],temp[4],temp[5])
                if not os.path.exists(os.path.join("artifacts_removed_data",temp[2])):
                    os.mkdir(os.path.join("artifacts_removed_data",temp[2]))
                if not os.path.exists(os.path.join("artifacts_removed_data",temp[2],temp[3])):
                    os.mkdir(os.path.join("artifacts_removed_data",temp[2],temp[3]))
                if not os.path.exists(os.path.join("artifacts_removed_data",temp[2],temp[3],temp[4])):
                    os.mkdir(os.path.join("artifacts_removed_data",temp[2],temp[3],temp[4]))
                try:
                    executor.submit(removeartifacts,f,os.path.join("artifacts_removed_data",temp[2],temp[3],temp[4],temp[5]))
                except Exception as e:
                    print("Exception occured: ",e)
        # if not os.path.exists(os.path.join("artifacts_removed_data",))
    if sys.argv[1]=="cb":
        if not os.path.exists("beta_extracted"):
            os.mkdir("beta_extracted")
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            for f in files:
                meta_data=getmetadata(f)
                subject_path=os.path.join("beta_extracted",meta_data["subject"])
                class_path=os.path.join(subject_path,meta_data["mov_class"])
                type_path=os.path.join(class_path,meta_data["ds_type"])
                if not os.path.exists(subject_path):
                    os.mkdir(subject_path)
                if not os.path.exists(class_path):
                    os.mkdir(class_path)
                if not os.path.exists(type_path):
                    os.mkdir(type_path)
                file_path=os.path.join("artifacts_removed_data",meta_data["subject"],meta_data["mov_class"],meta_data["ds_type"],f)
                save_path=os.path.join(type_path,f)
                try:
                    executor.submit(centralbetafrequency,file_path,save_path)
                except Exception as e:
                    print("Exception Occured ",e)
    if sys.argv[1]=="sp":
        refined_sample_path=refined_golay_filepaths[0]
        temp=refined_golay_filepaths[0].split("/")
        artifacts_removed_path=os.path.join("artifacts_removed_data",temp[2],temp[3],temp[4],temp[5])
        beta_extracted=os.path.join("beta_extracted",temp[2],temp[3],temp[4],temp[5])
        raw_data_path=os.path.join("reduced_data",temp[5])
        # print(refined_sample_path,artifacts_removed_path,beta_extracted)
        df1=pd.read_csv(raw_data_path,header=None)
        df2=pd.read_csv(refined_sample_path,header=None)
        df3=pd.read_csv(artifacts_removed_path,header=None)
        df4=pd.read_csv(beta_extracted,header=None)
        time_stamps=[i*15 for i in range(len(list(df1[:500][0])))]
        # print(len(time_stamps))
        for i in range(22):
            fig,a =  plt.subplots(4,sharex=True)
            fig.text(0.008, 0.55, 'Voltage', va='center', rotation='vertical')
            a[0].set_title("Raw Data")
            a[1].set_title("Golay Refined")
            a[2].set_title("Artifacts Removed")
            a[3].set_title("Beta_extracted")
            a[0].plot(time_stamps,list(df1[:500][i]))
            a[1].plot(time_stamps,list(df2[:500][i]))
            a[2].plot(time_stamps,list(df3[:500][i]))
            a[3].plot(time_stamps,list(df4[:500][i]))
            plt.xlabel("Time Stamps")
            plt.suptitle("Voltage Vs Time Stamps for Column "+str(i+1),ha='right',size='x-small')
            plt.tight_layout()
            plt.savefig("plots_compare_single_col"+str(i+1)+".tiff",format="tiff",dpi=350)
            plt.close()
        fig,a =  plt.subplots(4,sharex=True)
        fig.text(0.008, 0.55, 'Voltage', va='center', rotation='vertical')
        a[0].set_title("Raw Data")
        a[1].set_title("Golay Refined")
        a[2].set_title("Artifacts Removed")
        a[3].set_title("Beta_extracted")
        for i in range(len(df1.columns)):
            a[0].plot(time_stamps,list(df1[:500][i]),label="column-"+str(i+1))
            a[1].plot(time_stamps,list(df2[:500][i]),label="column-"+str(i+1))
            a[2].plot(time_stamps,list(df3[:500][i]),label="column-"+str(i+1))
            a[3].plot(time_stamps,list(df4[:500][i]),label="column-"+str(i+1))
        plt.xlabel("Time Stamps")
        plt.tight_layout()
        plt.savefig("plots_compare.tiff",format="tiff",dpi=350)
        plt.close()
    if sys.argv[1]=="ds":
        if not os.path.exists("segmented_data"):
            os.mkdir("segmented_data")
        for f in files:
            meta_data=getmetadata(f)
            subject_path=os.path.join("segmented_data",meta_data["subject"])
            class_path=os.path.join(subject_path,meta_data["mov_class"])
            type_path=os.path.join(class_path,meta_data["ds_type"])
            com_file_path=os.path.join(type_path,f[:-4])
            if not os.path.exists(subject_path):
                os.mkdir(subject_path)
            if not os.path.exists(class_path):
                os.mkdir(class_path)
            if not os.path.exists(type_path):
                os.mkdir(type_path)
            if not os.path.exists(com_file_path):
                os.mkdir(com_file_path)
        if not os.path.exists("min_n.pickle"):
            min_n=float('inf')
            pickle_out = open("min_n.pickle","wb")
            for f in channel_selected_file_paths:
                df=pd.read_csv(f,header=None)
                n=len(df[:][0])
                if(n<min_n):
                    min_n=n
            pickle.dump(min_n, pickle_out)
        else:
            p = open("min_n.pickle",'rb')
            min_n = pickle.load(p)
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            for f in channel_selected_file_paths:
                try:
                    executor.submit(segment_data,f)
                except Exception as e:
                    print("Exception Occured: ",e)
