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
    train_data[tot]+=vec
    tot+=1

np.save('./data/res_pos.npy',train_data)

pos=utils.read_neg_route()

train_data=np.zeros((len(pos),configs.HOG_size))

tot=0

for route in tqdm(pos):
    vec=utils.cal_HOG(route)
    train_data[tot]+=vec
    tot+=1

np.save('./data/res_neg.npy',train_data)
    