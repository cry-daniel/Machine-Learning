import matplotlib.pyplot as plt
#from numpy.lib.function_base import append
import configs
import cv2
import numpy as np

c_x=configs.cell_x
c_y=configs.cell_y
b_x=configs.block_x
b_y=configs.block_y

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
    res_=np.zeros(9)
    for i in range(pos_x*c_x,(pos_x+1)*c_x):
        for j in range(pos_y*c_y,(pos_y+1)*c_y):
            minn=int(arc_max[i][j]/20)
            res[minn]+=(20*(minn+1)-arc_max[i][j])/20*Img_max[i][j]
            res[min(minn+1,9)]+=(arc_max[i][j]-20*minn)/20*Img_max[i][j]
    res_=res[0:9]
    #print(res)
    res_[0]+=res[9]
    return res_

def NORM(res,pos_x,pos_y):
    a=[]
    for i in range(pos_x,pos_x+b_x):
        for j in range(pos_y,pos_y+b_y):
            for k in range(0,9):
                a.append(res[i][j][k])
    tot=0.
    for i in range(0,9*b_x*b_y):
        tot+=a[i]
    for i in range(0,9*b_x*b_y):
        a[i]/=tot
    return a

def show_pic(img,Img_y,Img_x,Img):
    plt.subplot(2,2,1)
    plt.title('origin')
    plt.imshow(img)
    #print(np.size(img,0),'x',np.size(img,1),'x',np.size(img,2))
    plt.subplot(2,2,2)
    plt.title('y position')
    plt.imshow(Img_y)
    
    plt.subplot(2,2,3)
    plt.title('x position')
    plt.imshow(Img_x)
    
    plt.subplot(2,2,4)
    plt.title('x * y')
    plt.imshow(Img)
    plt.show()

def cal_HOG(route):
    img=read_img(route)
    size = (64,128)
    img= cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    x=np.size(img,0)
    y=np.size(img,1)
    #print(x,y)
    Img_x=cv2.filter2D(img,cv2.CV_16S,configs.kernel_x)
    Img_y=cv2.filter2D(img,cv2.CV_16S,configs.kernel_y)
    Img=abs(Img_x)+abs(Img_y)
    #show_pic(img,Img_y,Img_x,Img)
    arc_max,Img_max=cal_arctan(Img,Img_x,Img_y)
    #print(arc_max)
    res=np.zeros([100,100,9])
    for i in range(0,int(x/configs.cell_x)):
        for j in range(0,int(y/configs.cell_y)):
            res[i][j]=histogram(Img_max,arc_max,i,j)  
    vec=[]
    for i in range(0,int(x/(configs.cell_x))-1):
        for j in range(0,int(y/(configs.cell_y))-1):
            a=NORM(res,i,j)
            for k in a:
                vec.append(k)
    if len(vec)!=configs.HOG_size:
        print("error!")
        input()
    return np.array(vec)