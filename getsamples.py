import os
import pandas as pd
import random
reduced_base_path="reduced_data"
sample_base_path="sample100"
files=sorted(list(os.listdir(reduced_base_path)))

def getmetadata(f):
    temp=list(f.split("."))
    name=temp[0]
    subject=name[2]
    mov_class=name[-1]
    ds_type=name[3]
    meta_data={"subject":subject,"mov_class":mov_class,"ds_type":ds_type}
    return meta_data

def getintervals(n,size=100):#returns 4 random intervals of size 100
    a=int(n/4)
    try:
        i1=random.randint(0,a-size-1)
        i2=random.randint(a,2*a-size-1)
        i3=random.randint(2*a,3*a-size-1)
        i4=random.randint(3*a,n-size-1)
    except Exception as e:
        print(e)
    l=[]
    l.append([i1,i1+size])
    l.append([i2,i2+size])
    l.append([i3,i3+size])
    l.append([i4,i4+size])
    return l



for f in files:
    meta_data=getmetadata(f)
    subject_path=os.path.join(sample_base_path,"subject1")
    class_path=os.path.join(subject_path,meta_data["mov_class"])
    type_path=os.path.join(class_path,meta_data["ds_type"])
    if not os.path.exists(sample_base_path):
        os.mkdir(sample_base_path)
    if meta_data["subject"]=='1':
        if not os.path.exists(subject_path):
            os.mkdir(subject_path)
        if not os.path.exists(class_path):
            os.mkdir(class_path)
        if not os.path.exists(type_path):
            os.mkdir(type_path)
        temp=list(f.split("."))
        file_path=os.path.join(reduced_base_path,f)
        df=pd.read_csv(file_path,header=None)
        row_count=len(df)
        intervals=getintervals(row_count,100)
        print("For File ",f,"Rows count are ",row_count,"And Intervals ",intervals )
        for i in range(len(intervals)):
            save_path=os.path.join(type_path,temp[0]+"part_"+str(i+1)+"."+temp[1])
            temp_df=df[intervals[i][0]:intervals[i][1]]
            temp_df.to_csv(save_path,index=False,header=None)