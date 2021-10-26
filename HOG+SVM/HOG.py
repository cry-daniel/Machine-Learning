import utils
import configs
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm

pos=utils.read_pos_route()

train_data=np.zeros((len(pos),configs.HOG_size))

tot=0

for route in tqdm(pos):
    vec=utils.cal_HOG(route)
    #print(vec)
    #print(np.size(train_data,0),np.size(train_data,1))
    train_data[tot]=vec
    tot+=1

np.save('./data/res_pos.npy',train_data)

neg=utils.read_neg_route()

train_data=np.zeros((10*len(neg),configs.HOG_size))

tot=0

for route in tqdm(neg):
    vec=utils.cal_neg_HOG(route)
    #print(vec)
    #print(np.size(vec,0),np.size(vec,1),np.size(vec,2))
    train_data[tot*10:(tot+1)*10]=vec[:,0,:]
    #print(train_data)
    tot+=1

print(np.size(train_data,0),np.size(train_data,1))

np.save('./data/res_neg.npy',train_data)