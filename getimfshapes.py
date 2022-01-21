import os
import pickle
files=os.listdir("imfs")
shapes={}
for f in files:
    p = open(os.path.join("imfs",f),'rb')
    imf = pickle.load(p)
    print("Working on ",f)
    if imf.shape[0] in shapes.keys():
        shapes[imf.shape[0]]+=1
    else:
        shapes[imf.shape[0]]=1
    print(shapes)

f = open("demofile2.txt", "w")
f.write(str(shapes))
f.close()

