import matplotlib.pyplot as plt
import configs
import cv2
import numpy as np
from sklearn import svm
import pickle
from tqdm import tqdm
import random

c_x=configs.cell_x
c_y=configs.cell_y
b_x=configs.block_x
b_y=configs.block_y

def read_pos_route():   #读取正确训练图片路径（最后）
    pos=[]
    fp=open(configs.pos_route,'r')
    for line in fp.readlines():
        line=line.strip()
        pos.append(line)
    return pos

def read_neg_route():   #读取错误训练图片路径（最后）
    neg=[]
    fp=open(configs.neg_route,'r')
    for line in fp.readlines():
        line=line.strip()
        neg.append(line)
    return neg

def read_img(route):    #读图片
    img=cv2.imread('./INRIAPerson/'+route)
    return img

def cal_arctan(Img,Img_x,Img_y):    #计算反正切和选模最长
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
            #算arctan
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

def histogram(Img_max,arc_max,pos_x,pos_y):     #做直方图
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

def NORM(res,pos_x,pos_y):  #归一化
    a=[]
    for i in range(pos_x,pos_x+b_x):
        for j in range(pos_y,pos_y+b_y):
            for k in range(0,9):
                a.append(res[i][j][k])
    tot=0.
    for i in range(0,9*b_x*b_y):
        tot+=a[i]
    for i in range(0,9*b_x*b_y):
        if tot>0:
            a[i]/=tot
    return a

def show_pic(img,Img_y,Img_x,Img):  #显示图片，顺序：原图，Y偏导，X偏导，梯度
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



def read_test_pos_route():   #读取正确训练图片路径（最后）
    pos=[]
    fp=open(configs.test_pos_route,'r')
    for line in fp.readlines():
        line=line.strip()
        pos.append(line)
    return pos

def read_test_neg_route():   #读取错误训练图片路径（最后）
    neg=[]
    fp=open(configs.test_neg_route,'r')
    for line in fp.readlines():
        line=line.strip()
        neg.append(line)
    return neg

def svm_train(training_data,training_label):
    #clf = svm.SVC(C=1000, kernel='rbf', gamma=0.1, decision_function_shape='ovr')
    #clf=svm.SVR(C=10000,epsilon=1e-6)
    clf = svm.SVC(kernel="linear", probability=True)
    #clf=svm.LinearSVC(C=10000,max_iter=25000)
    clf.fit(training_data,training_label.ravel())
    return clf

def svm_predict(clf,test_data):
    test_label=clf.predict(test_data)
    return test_label

def svm_save(clf):
    s=pickle.dumps(clf)
    f=open('./data/svm.model','wb+')
    f.write(s)
    f.close
    print('Save done!')

def svm_load(route):
    f2=open(route,'rb')
    s2=f2.read()
    clf=pickle.loads(s2)
    return clf

def save_test_data(route):
    length=len(route)
    test_data=np.zeros((length,configs.HOG_size))
    tot=0
    for i in tqdm(route):
        img=cal_HOG(i)
        np.array([img])
        test_data[tot]=img
        #print(test_data[tot])
        tot+=1
        #input()
    test_data=np.array(test_data,dtype='float32')
    #print(test_data)
    np.save('./data/test_data.npy',test_data)


def check_pos_or_neg(clf,test_data):
    print('Starting predicting!')
    par=svm_predict(clf,test_data)
    print('Prediction finish!')
    return par

def cal_HOG(route):     #计算HOG，返回值为大小为configs.HOG_size的向量
    img=read_img(route)
    size = (64,128)
    img= cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    #print(np.size(img,0),np.size(img,1),np.size(img,2))
    hog = cv2.HOGDescriptor((64, 128), (16, 16), (8, 8), (8, 8), 9)
    return np.array(hog.compute(img).T)

def cal_neg_HOG(route):     #计算HOG，返回值为大小为configs.HOG_size的向量
    img=read_img(route)
    (length,width)=[np.size(img,0),np.size(img,1)]
    ans=[]
    hog = cv2.HOGDescriptor((64, 128), (16, 16), (8, 8), (8, 8), 9)
    for l in range(10):
        #print(l)
        rand0 = random.randint(0, np.size(img,0) - 128)
        rand1 = random.randint(0, np.size(img,1) - 64)
        Img = img[rand0:rand0 + 128, rand1:rand1 + 64, ::-1]
        '''
        plt.imshow(Img)
        plt.show()
        #print(np.size(img,0),np.size(img,1),np.size(img,2))
        '''
        ans.append(hog.compute(Img).T)
    return np.array(ans)
    #print(ans)
    #input()

