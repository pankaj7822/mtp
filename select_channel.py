import pandas as pd
import os
import concurrent.futures
root_path=""
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
  channel_to_be_selected=[8, 10, 12, 16, 14, 22, 3, 19]
  l=[i-1 for i in channel_to_be_selected]
  final_df = df.iloc[:,l]
  final_df.to_csv(save_path,index=False,header=None)



## 8, 10, 12, 16, 14, 22, 3, 19, 15, 5, 13, 11, 21, 9, 
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