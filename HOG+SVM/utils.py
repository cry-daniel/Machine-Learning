import configs
import cv2
import numpy as np

def read_pos_route():
    pos=[]
    fp=open(configs.pos_route,'r')
    for line in fp.readlines():
        line=line.strip()
        pos.append(line)
    return pos

def read_neg_route():
    neg=[]
    fp=open(configs.neg_route,'r')
    for line in fp.readlines():
        line=line.strip()
        neg.append(line)
    return neg

def read_img(route):
    img=cv2.imread('./INRIAPerson/'+route)
    return img