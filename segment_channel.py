import pandas as pd
import os
import concurrent.futures
import sys

root_path=""
min_n=144210
beta_extracted_base_path=os.path.join(root_path,"beta_extracted")
channel_selected_base_path=os.path.join(root_path,"channel_selected")
segmented_data_base_path=os.path.join(root_path,"segmented_data")
channel_selected_file_paths=[]
beta_extracted_filepaths=[]
reduced_base_path=os.path.join(root_path,"reduced_data")
files=sorted(list(os.listdir(reduced_base_path)))
for i in range(1,10):
    for j in range(1,5):
        for t in ["E","T"]:
          file_name="A0"+str(i)+t+"_"+str(j)+".csv"
          r=os.path.join("beta_extracted",str(i),str(j),str(t),file_name)
          beta_extracted_filepaths.append(r)
          s=os.path.join("channel_selected",str(i),str(j),str(t),file_name)
          channel_selected_file_paths.append(s)

print(beta_extracted_filepaths)
print(channel_selected_file_paths)
print(files)


def selectchannel(file_path,save_path):
  print("selected channels",file_path)
  df=pd.read_csv(file_path,header=None)
  channel_to_be_selected=[8, 10, 12, 16, 14, 22, 3, 19, 15, 5, 13, 11, 21, 9]
  l=[i-1 for i in channel_to_be_selected]
  final_df = df.iloc[:,l]
  final_df.to_csv(save_path,index=False,header=None)

def getmetadata(f):
  temp=list(f.split("."))
  name=temp[0]
  subject=name[2]
  mov_class=name[-1]
  ds_type=name[3]
  meta_data={"subject":subject,"mov_class":mov_class,"ds_type":ds_type}
  return meta_data

def segment_data(file_path):
    print("Segmenting ",file_path)
    temp=file_path.split("/")
    df=pd.read_csv(os.path.join(root_path,file_path),header=None)
    start=0
    end=min_n
    count=0
    for i in range(int(min_n/1000)):
        count+=1
        j=start
        k=min(j+1000,end)
        temp_df=df[j:k]
        temp_df.to_csv(os.path.join(segmented_data_base_path,temp[1],temp[2],temp[3],temp[4][:-4],temp[4][:-4]+"part_"+str(count)+".csv"),header=None,index=False)
        start+=1000
    print("Segmented ",file_path)

if __name__ == "__main__":
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
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            for f in channel_selected_file_paths:
                try:
                    executor.submit(segment_data,f)
                except Exception as e:
                    print("Exception Occured: ",e)

    if sys.argv[1]=="cs":
        if not os.path.exists(channel_selected_base_path):
            os.mkdir(channel_selected_base_path)
            with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
                for f in beta_extracted_filepaths:
                    temp=f.split("/")
                    save_path=os.path.join(channel_selected_base_path,temp[1],temp[2],temp[3],temp[4])
                    if not os.path.exists(os.path.join(channel_selected_base_path,temp[1])):
                        os.mkdir(os.path.join(channel_selected_base_path,temp[1]))
                    if not os.path.exists(os.path.join(channel_selected_base_path,temp[1],temp[2])):
                        os.mkdir(os.path.join(channel_selected_base_path,temp[1],temp[2]))
                    if not os.path.exists(os.path.join(channel_selected_base_path,temp[1],temp[2],temp[3])):
                        os.mkdir(os.path.join(channel_selected_base_path,temp[1],temp[2],temp[3]))
                    try:
                        executor.submit(selectchannel,os.path.join(root_path,f),os.path.join(channel_selected_base_path,temp[1],temp[2],temp[3],temp[4]))
                    except Exception as e:
                        print("Exception occured: ",e)