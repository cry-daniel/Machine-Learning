import matplotlib
import utils
import numpy as np
import cv2
import matplotlib.pyplot as plt 
from tqdm import tqdm
from scipy import interpolate

pos=utils.read_test_pos_route()
neg=utils.read_test_neg_route()

#print(len(pos))

test_label=[]
test_route=[]

def init_xy(test_data,clf,length):
    x=[]
    y=[]
    FPPW=[]
    #par=utils.check_pos_or_neg(clf,test_data)
    for i in tqdm(range(0,100,5)):
        par=clf.predict_proba(test_data)
        maxx=i/100
        temp=[]
        for i in par:
            if i[1]>maxx:
                temp.append(1)
            else:
                temp.append(-1)
        right=tot=0
        AUC=np.zeros((2,2)) #AUC 0,0 Tn ; 1,0 FP ; 0,1 FN ; 1,1 TP
        for i in range(0,len(temp)):
            if temp[i]==test_label[i]:
                right+=1
            AUC[max(0,temp[i])][max(0,test_label[i])]+=1
            tot+=1
        x.append(AUC[1][0]/(AUC[1][0]+AUC[0][0]))
        y.append(AUC[1][1]/(AUC[1][1]+AUC[0][1]))
        #print(AUC[1][0],length)
        fppw=(AUC[1][0])/length
        FPPW.append(fppw)
    temp=0
    x_=[]
    y_=[]
    miss_=[]
    FPPW_=[]
    for i in range(len(x)):
        if temp==x[i]:
            continue
        else:
            x_.append(x[i])
            y_.append(y[i])
            miss_.append(1-y[i])
            FPPW_.append(FPPW[i])
            temp=x[i]
    x=x_
    x.append(0.)
    y=y_
    y.append(0.)
    miss=np.array(miss_)
    FPPW=np.array(FPPW_)
    x=np.array(x)
    y=np.array(y)
    np.save('./data/x.npy',x)
    np.save('./data/y.npy',y)
    np.save('./data/miss.npy',miss)
    np.save('./data/FPPW.npy',FPPW)

    return x,y,miss,FPPW


for i in pos:
    test_label.append(1)
    test_route.append(i)

for i in neg:
    test_label.append(-1)
    test_route.append(i)

np.array(test_label)

length=np.size(test_label)

clf=utils.svm_load('./data/svm.model')

try:
    f = open("./data/test_data.npy")
    f.close()
    test_data=np.load('./data/test_data.npy')
    print('Testdata Find')
except IOError:
    print('Testdata Not find')      
    utils.save_test_data(test_route)
    test_data=np.load('./data/test_data.npy')
    
try:
    f=open('./data/x.npy')
    f.close()
    f=open('./data/y.npy')
    f.close()
    f=open('./data/miss.npy')
    f.close()
    f=open('./data/FPPW.npy')
    f.close()
    x=np.load('./data/x.npy')
    y=np.load('./data/y.npy')
    miss=np.load('./data/miss.npy')
    FPPW=np.load('./data/FPPW.npy')
    print('X & Y & miss & FPPW Find')
except:
    print('X & Y & miss & FPPW not Find')
    x,y,miss,FPPW=init_xy(test_data,clf,length)

def draw_AUC(x,y):
    x_new=np.linspace(0,1,101)
    f=interpolate.interp1d(x,y,kind='slinear')
    y_new=f(x_new)
    tot=0
    for i in range(0,np.size(x_new)-2):
        tot+=(y_new[i]+y_new[i+1])*(x_new[i+1]-x_new[i])/2

    print(tot)

    #plt.plot(x_new,y_new)
    plt.xlabel('FPRate')
    plt.ylabel('TPRate')
    plt.title('AUC')
    plt.plot(x,y,'*')
    #plt.plot(x,y)
    plt.show()

#draw_AUC(x,y)

def draw_DET(miss,FPPW):
    #print(FPPW)
    miss=np.log10(miss)
    FPPW=np.log10(FPPW)
    #plt.axis([0.001,0.3,0.01,0.6])
    #plt.xscale('log')
    #plt.yscale('log')
    #my_x_ticks=
    plt.title('DET - based on INRIA database')
    plt.xlabel('log ( false positives per window (FPPW) )')
    plt.ylabel('log ( miss rate )')
    plt.plot(FPPW,miss,'*')
    plt.show()

draw_DET(miss,FPPW)