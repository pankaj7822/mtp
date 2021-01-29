import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
import os
import sys
import concurrent.futures
from scipy.stats import entropy

sample_path="sample100"
sample_data_base_path=os.path.join(sample_path,"subject1")
visualization_base_path=os.path.join(sample_path,"visualizations")
refined_base_path=os.path.join(sample_path,"refined_subject1")
types=["E","T"]

def getfilepath(base_path):
    subject_paths=[]
    file_paths=[]
    type_paths=[]
    subject_paths.append(os.path.join(base_path,"subject1"))
    for s in subject_paths:
        for i in range(1,5):
            for t in types:
                type_paths.append(os.path.join(s,str(i),t))
    for t_path in type_paths:
        temp2=sorted(list(os.listdir(t_path)))
        for t in temp2:
            file_paths.append(os.path.join(t_path,t))
    return file_paths

def signaltonoise(a, axis=0, ddof=0):
    a = np.asanyarray(a)
    m = a.mean(axis)
    sd = a.std(axis=axis, ddof=ddof)
    return np.where(sd == 0, 0, m/sd)        

def getmetadata(f):
    temp=list(f.split("."))
    name=temp[0]
    subject=name[2]
    mov_class=name[-1]
    ds_type=name[3]
    meta_data={"subject":subject,"mov_class":mov_class,"ds_type":ds_type}
    return meta_data

def refine_data_butterworth(data_file_path,save_path,fs=250,fc=100):
    df=pd.read_csv(data_file_path,header=None)
    df2=pd.DataFrame()
    for i in range(len(df.columns)):
        data=list(df[:][i]) # Generate data properly  
        w = fc / (fs / 2)  # Normalize the frequency
        b, a = signal.butter(5, w, 'low') #5th order low banpass filter
        butterworth_output = signal.filtfilt(b, a, data)
        df2[i]=butterworth_output
    df2.to_csv(save_path,index=False,header=None)
    print("Butterworth Refined Data for ",data_file_path)    

        
def refine_data_golay(data_file_path,save_path,window_size=7,poly_order=3):
    df=pd.read_csv(data_file_path,header=None)
    df2=pd.DataFrame()
    for i in range(len(df.columns)):
        data=list(df[:][i]) # Generate data properly  
        golay_output= signal.savgol_filter(data, window_size, poly_order)
        df2[i]=golay_output
    df2.to_csv(save_path,index=False,header=None)
    print("Golay Refined Data for ",data_file_path)

def calculate_entropy(dataframe_path):
    df=pd.read_csv(dataframe_path,header=None)
    avg_entropy=0.0
    for i in range(len(df.columns)):
        data=df[:][i].value_counts()
        cur_entropy=entropy(data)
        avg_entropy+=cur_entropy
    avg_entropy=avg_entropy/float(len(df.columns))
    return avg_entropy


def calculate_sn_ratio(dataframe_path):
    df=pd.read_csv(dataframe_path,header=None)
    avg_snr=0.0
    for i in range(len(df.columns)):
        data=df[:][i]
        snr=signaltonoise(data)
        avg_snr+=snr
    avg_snr=avg_snr/float(len(df.columns))
    return avg_snr


time_stamps=[i for i in range(0,100)]    
file_paths=getfilepath(sample_path)
    
if __name__ == "__main__":
    if sys.argv[1]=="rf":
        if not os.path.exists(refined_base_path):
            os.mkdir(refined_base_path)
        butterworth_path=os.path.join(refined_base_path,"butter_worth")
        golay_path=os.path.join(refined_base_path,"golay")
        if not os.path.exists(butterworth_path):
            os.mkdir(butterworth_path)
        if not os.path.exists(golay_path):
            os.mkdir(golay_path)
        with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
            for f in file_paths:
                temp=f[-16:-10]
                meta_data=getmetadata(temp)
                if meta_data["subject"]=='1':
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
                        save_path1=os.path.join(type_path,f[-16:])
                        save_path2=os.path.join(type_path2,f[-16:])
                        executor.submit(refine_data_butterworth,f,save_path1)
                        executor.submit(refine_data_golay,f,save_path2)
                    except Exception as e:
                        print("Exception: ",e)
    
    if sys.argv[1]=="v":
        if not os.path.exists(visualization_base_path):
            os.mkdir(visualization_base_path)
        for f in file_paths:
            temp=list(f.split("/"))
            print(temp)
            g_path=os.path.join(temp[0],"refined_subject1","golay","1",temp[2],temp[3],temp[4])
            b_path=os.path.join(temp[0],"refined_subject1","butter_worth","1",temp[2],temp[3],temp[4])
            raw_df=pd.read_csv(f,header=None)
            g_df=pd.read_csv(g_path,header=None)
            b_df=pd.read_csv(b_path,header=None)
            raw_visual_path=os.path.join(visualization_base_path,"raw_visulazations")
            golay_visual_path=os.path.join(visualization_base_path,"golay_visulazations")
            butterworth_visual_path=os.path.join(visualization_base_path,"butterworth_visulazations")
            combined_visual_path=os.path.join(visualization_base_path,"combined_visulazations")
            
            if not os.path.exists(raw_visual_path):
                os.mkdir(raw_visual_path)
            if not os.path.exists(golay_visual_path):
                os.mkdir(golay_visual_path)
            if not os.path.exists(butterworth_visual_path):
                os.mkdir(butterworth_visual_path)
            if not os.path.exists(combined_visual_path):
                os.mkdir(combined_visual_path)
                
                
            if not os.path.exists(os.path.join(raw_visual_path,temp[2])):
                os.mkdir(os.path.join(raw_visual_path,temp[2]))
            if not os.path.exists(os.path.join(golay_visual_path,temp[2])):
                os.mkdir(os.path.join(golay_visual_path,temp[2]))
            if not os.path.exists(os.path.join(butterworth_visual_path,temp[2])):
                os.mkdir(os.path.join(butterworth_visual_path,temp[2]))
            if not os.path.exists(os.path.join(combined_visual_path,temp[2])):
                os.mkdir(os.path.join(combined_visual_path,temp[2]))
            
            if not os.path.exists(os.path.join(raw_visual_path,temp[2],temp[3])):
                os.mkdir(os.path.join(raw_visual_path,temp[2],temp[3]))
            if not os.path.exists(os.path.join(golay_visual_path,temp[2],temp[3])):
                os.mkdir(os.path.join(golay_visual_path,temp[2],temp[3]))
            if not os.path.exists(os.path.join(butterworth_visual_path,temp[2],temp[3])):
                os.mkdir(os.path.join(butterworth_visual_path,temp[2],temp[3]))
            if not os.path.exists(os.path.join(combined_visual_path,temp[2],temp[3])):
                os.mkdir(os.path.join(combined_visual_path,temp[2],temp[3]))
            
            if not os.path.exists(os.path.join(raw_visual_path,temp[2],temp[3],temp[4][:-4])):
                os.mkdir(os.path.join(raw_visual_path,temp[2],temp[3],temp[4][:-4]))
            if not os.path.exists(os.path.join(golay_visual_path,temp[2],temp[3],temp[4][:-4])):
                os.mkdir(os.path.join(golay_visual_path,temp[2],temp[3],temp[4][:-4]))
            if not os.path.exists(os.path.join(butterworth_visual_path,temp[2],temp[3],temp[4][:-4])):
                os.mkdir(os.path.join(butterworth_visual_path,temp[2],temp[3],temp[4][:-4]))
            if not os.path.exists(os.path.join(combined_visual_path,temp[2],temp[3],temp[4][:-4])):
                os.mkdir(os.path.join(combined_visual_path,temp[2],temp[3],temp[4][:-4]))
            
            if sys.argv[2]=="r":
                for j in range(len(raw_df.columns)):
                    plt.plot(time_stamps,raw_df[:][j],label="Raw Visualization")
                    plt.xlabel("Time Stamps")
                    plt.ylabel("Voltage")
                    plt.title('Voltage Vs Time Stamps')
                    raw_save_path=os.path.join(raw_visual_path,temp[2],temp[3],temp[4][:-4],"column"+str(j+1)+".tiff")
                    plt.savefig(raw_save_path,format="tiff",dpi=350)
                    plt.close()
                    print("Plotted for ",raw_save_path,"column ",j)
                for k in range(len(raw_df.columns)):
                    plt.plot(time_stamps,raw_df[:][k],label=k+1)
                    plt.xlabel("Time Stamps")
                    plt.ylabel("Voltage")
                    plt.title('Voltage Vs Time Stamps',loc='left')
                    plt.legend(loc='upper right',columnspacing=0.3,bbox_to_anchor=(1.10,1.15),
                        ncol=11, fancybox=True, labelspacing=0.3,handlelength=0.4,handletextpad=0.1)
                    m_raw_save_path=os.path.join(raw_visual_path,temp[2],temp[3],temp[4][:-4],"combined.tiff")
                    plt.savefig(m_raw_save_path,format="tiff",dpi=350)
                plt.close()
                print("Plotted for ",m_raw_save_path)
            
            if sys.argv[2]=="g":
                for j in range(len(g_df.columns)):
                    plt.plot(time_stamps,g_df[:][j],label="Golay Visualization")
                    plt.xlabel("Time Stamps")
                    plt.ylabel("Voltage")
                    plt.title('Voltage Vs Time Stamps')
                    golay_save_path=os.path.join(golay_visual_path,temp[2],temp[3],temp[4][:-4],"column"+str(j+1)+".tiff")
                    plt.savefig(golay_save_path,format="tiff",dpi=350)
                    plt.close()
                    print("Plotted for ",golay_save_path,"column ",j)

                for k in range(len(g_df.columns)):
                    plt.plot(time_stamps,g_df[:][k],label=k+1)
                    plt.xlabel("Time Stamps")
                    plt.ylabel("Voltage")
                    plt.title('Voltage Vs Time Stamps',loc='left')
                    plt.legend(loc='upper right',columnspacing=0.3,bbox_to_anchor=(1.10,1.15),
                        ncol=11, fancybox=True, labelspacing=0.3,handlelength=0.4,handletextpad=0.1)
                    m_golay_save_path=os.path.join(golay_visual_path,temp[2],temp[3],temp[4][:-4],"combined.tiff")
                    plt.savefig(m_golay_save_path,format="tiff",dpi=350)
                plt.close()
                print("Plotted for ",m_golay_save_path)
            
            if sys.argv[2]=="b":
                for j in range(len(b_df.columns)):
                    plt.plot(time_stamps,b_df[:][j],label="Golay Visualization")
                    plt.xlabel("Time Stamps")
                    plt.ylabel("Voltage")
                    plt.title('Voltage Vs Time Stamps')
                    butter_save_path=os.path.join(butterworth_visual_path,temp[2],temp[3],temp[4][:-4],"column"+str(j+1)+".tiff")
                    plt.savefig(butter_save_path,format="tiff",dpi=350)
                    plt.close()
                    print("Plotted for ",butter_save_path,"column ",j)

                for k in range(len(g_df.columns)):
                    plt.plot(time_stamps,b_df[:][k],label=k+1)
                    plt.xlabel("Time Stamps")
                    plt.ylabel("Voltage")
                    plt.title('Voltage Vs Time Stamps',loc='left')
                    plt.legend(loc='upper right',columnspacing=0.3,bbox_to_anchor=(1.10,1.15),
                        ncol=11, fancybox=True, labelspacing=0.3,handlelength=0.4,handletextpad=0.1)
                    m_butter_save_path=os.path.join(butterworth_visual_path,temp[2],temp[3],temp[4][:-4],"combined.tiff")
                    plt.savefig(m_butter_save_path,format="tiff",dpi=350)
                plt.close()
                print("Plotted for ",m_butter_save_path)
            
            if sys.argv[2]=="c":
                for j in range(len(b_df.columns)):
                    plt.plot(time_stamps,g_df[:][j],label="Golay Visualization")
                    plt.plot(time_stamps,b_df[:][j],label="Butterworth Visualization")
                    plt.plot(time_stamps,raw_df[:][j],label="Raw Visualization")
                    plt.xlabel("Time Stamps")
                    plt.ylabel("Voltage")
                    plt.title('Voltage Vs Time Stamps')
                    combined_save_path=os.path.join(combined_visual_path,temp[2],temp[3],temp[4][:-4],"column"+str(j+1)+".tiff")
                    plt.legend()
                    plt.savefig(combined_save_path,format="tiff",dpi=350)
                    plt.close()
                    print("Plotted for ",combined_save_path,"column ",j)

                # for k in range(len(g_df.columns)):
                #     plt.plot(time_stamps,g_df[:][k],label=k+1)
                #     plt.xlabel("Time Stamps")
                #     plt.ylabel("Voltage")
                #     plt.title('Voltage Vs Time Stamps',loc='left')
                #     plt.legend(loc='upper right',columnspacing=0.3,bbox_to_anchor=(1.10,1.15),
                #         ncol=11, fancybox=True, labelspacing=0.3,handlelength=0.4,handletextpad=0.1)
                #     m_butter_save_path=os.path.join(butterworth_visual_path,temp[2],temp[3],temp[4][:-4],"combined.tiff")
                #     plt.savefig(m_butter_save_path,format="tiff",dpi=350)
                # plt.close()
                # print("Plotted for ",m_butter_save_path)
            
            # plt.plot(time_stamps,raw_df[:][0],label="Unfiltered")
            # plt.plot(time_stamps,g_df[:][0],label="Golay Filtered")
            # plt.plot(time_stamps,b_df[:][0],label="Butterworth Filtered")

            # plt.legend()
            # plt.show()
            
    if sys.argv[1]=="e":
        entropies=[]
        snrs=[]
        for f in file_paths:
            temp=list(f.split("/"))
            g_path=os.path.join(temp[0],"refined_subject1","golay","1",temp[2],temp[3],temp[4])
            b_path=os.path.join(temp[0],"refined_subject1","butter_worth","1",temp[2],temp[3],temp[4])
            entropy_data={"File_Name":"","Raw_Entropy":0,"Golay_Entropy":0,"Butterworth_Entropy":0}
            snr_data={"File_Name":"","Raw_SNR":0,"Golay_SNR":0,"Butterworth_SNR":0}
            if not os.path.exists(os.path.join(sample_path,"entropy")):
                os.mkdir(os.path.join(sample_path,"entropy"))
            if not os.path.exists(os.path.join(sample_path,"snr")):
                os.mkdir(os.path.join(sample_path,"snr"))
            entropy_data["File_Name"]= temp[-1]
            entropy_data["Raw_Entropy"]=calculate_entropy(f)
            entropy_data["Golay_Entropy"]=calculate_entropy(g_path)   
            entropy_data["Butterworth_Entropy"]=calculate_entropy(b_path)
            snr_data["File_Name"]=temp[-1]
            snr_data["Raw_SNR"]=calculate_sn_ratio(f)
            snr_data["Golay_SNR"]=calculate_sn_ratio(g_path)
            snr_data["Butterworth_SNR"]=calculate_sn_ratio(b_path)   
            entropies.append(entropy_data)
            snrs.append(snr_data)
        e_df=pd.DataFrame(entropies)
        snr_df=pd.DataFrame(snrs)
        e_df.to_csv(os.path.join(sample_path,"entropy","Entropy.csv"))
        snr_df.to_csv(os.path.join(sample_path,"snr","snr.csv"))
        re_data=e_df[:]["Raw_Entropy"]
        ge_data=e_df[:]["Golay_Entropy"]
        be_data=e_df[:]["Butterworth_Entropy"]
        rsnr_data=snr_df[:]["Raw_SNR"]
        gsnr_data=snr_df[:]["Golay_SNR"]
        bsnr_data=snr_df[:]["Butterworth_SNR"]
        stamps=[i for i in range(len(e_df))]
        
        plt.plot(stamps,re_data,label="Raw Data Entropy")
        plt.plot(stamps,ge_data,label="Golay Refined Data Entropy")
        plt.plot(stamps,be_data,label="Butterworth Refined Data Entropy")
        plt.ylabel("Average Entropy")
        plt.xlabel("Files")
        plt.title('Average Entropy Vs Files',loc='left')
        plt.legend(loc='upper right',bbox_to_anchor=(1.10,1.15),fancybox=True,labelspacing=0.3,handlelength=0.4,handletextpad=0.1)
        plt.savefig(os.path.join(sample_path,"entropy","entropyplot.tiff"),format="tiff",dpi=350)
        plt.close()
        
        plt.plot(stamps,rsnr_data,label="Raw Data SNR")
        plt.plot(stamps,gsnr_data,label="Golay Refined Data SNR")
        plt.plot(stamps,bsnr_data,label="Butterworth Refined Data SNR")
        plt.ylabel("Average SNR")
        plt.xlabel("Files")
        plt.title('Average SNR Vs Files',loc='left')
        plt.legend(loc='upper right',bbox_to_anchor=(1.10,1.15),fancybox=True,labelspacing=0.3,handlelength=0.4,handletextpad=0.1)
        plt.savefig(os.path.join(sample_path,"snr","snrplot.tiff"),format="tiff",dpi=350)
        plt.close()
            