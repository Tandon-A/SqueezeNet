import os 
import random 
from PIL import Image
import numpy as np 

#Used tiny Imagenet Dataset for Training. 

tr_file = "squeeze\\dataset\\train_file1.txt"
val_file = "squeeze\\dataset\\val_file1.txt"
path = "squeeze\\dataset\\tiny-imagenet-200\\"
files = os.listdir(path + "train\\")
data = []

for i in range(len(files)):
    imgs = os.listdir(path +"train\\" + str(files[i]) +"\\images\\")
    for j in imgs:
        path_img = str(path + "train\\" + str(files[i]) + "\\images\\" + str(j))
        img = np.array(Image.open(path_img))
        #checking if image is grayscale or not
        if len(img.shape) == 3:
            data.append(str(path_img+" "+str(i)+"\n"))
    
print (len(data))
random.shuffle(data)

with open(tr_file,"w") as f:
    for i in data:
        f.write(i)


data =[]
with open("squeeze\\dataset\\tiny-imagenet-200\\val\\val_annotations.txt") as f:
    data = f.readlines()

data2= []
for i in data:
    line = i.split('\t')
    img = line[0]
    path_img = str(path+"val\\images\\" + img)
    img = np.array(Image.open(path_img))
    #checking if image is grayscale or not 
    if len(img.shape) == 3:
        cl = line[1]
        for j in range(len(files)):
            #getting label for validation image from val_annotations.txt
            if str(cl) == str(files[j]):
                pred = j
                break
        data2.append(str(path_img+" "+str(pred)+"\n")) 

random.shuffle(data2)

with open(val_file,"w") as f:
    for i in data2:
        f.write(i)
        
