import configs
import cv2
import numpy as np

c_x=configs.cell_x
c_y=configs.cell_y

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

def cal_arctan(Img,Img_x,Img_y):
    length=np.size(Img,0)
    width=np.size(Img,1)
    temp=np.zeros((length,width))
    temp_img=np.zeros((length,width))
    for i in range(0,length):
        for j in range(0,width):
            maxx=-1
            flag=-1
            for k in range(0,3):
                if Img[i][j][k]>maxx:
                    maxx=Img[i][j][k]
                    flag=k
            temp_img[i][j]=Img[i][j][flag]
            if Img_x[i][j][flag]==0:
                if Img_y[i][j][flag]==0:
                    temp[i][j]=0
                else:
                    temp[i][j]=90
                    
            elif Img_y[i][j][flag]==0:
                if temp[i][j]<0:
                    temp[i][j]=0
                else:
                    temp[i][j]=180
            else:
                temp[i][j]=abs(np.arctan2(Img_x[i][j][flag],Img_y[i][j][flag])*180/np.pi)        
    return temp,temp_img

def histogram(Img_max,arc_max,pos_x,pos_y):
    length=np.size(Img_max,0)
    width=np.size(Img_max,1)
    res=np.zeros(10)
    for i in range(pos_x*c_x,(pos_x+1)*c_x):
        for j in range(pos_y*c_y,(pos_y+1)*c_y):
            minn=int(arc_max[i][j]/20)
            res[minn]+=(20*(minn+1)-arc_max[i][j])/20*Img_max[i][j]
            res[min(minn+1,9)]+=(arc_max[i][j]-20*minn)/20*Img_max[i][j]
    return res