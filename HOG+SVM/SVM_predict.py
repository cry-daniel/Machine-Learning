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

def init_xy(test_data,clf):
    x=[]
    y=[]
    #par=utils.check_pos_or_neg(clf,test_data)
    for i in tqdm(range(0,100,1)):
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
    temp=0
    x_=[]
    y_=[]
    for i in range(len(x)):
        if temp==x[i]:
            continue
        else:
            x_.append(x[i])
            y_.append(y[i])
            temp=x[i]
    x=x_
    x.append(0.)
    y=y_
    y.append(0.)
    x=np.array(x)
    y=np.array(y)
    np.save('./data/x.npy',x)
    np.save('./data/y.npy',y)
    return x,y


for i in pos:
    test_label.append(1)
    test_route.append(i)

for i in neg:
    test_label.append(-1)
    test_route.append(i)

np.array(test_label)

clf=utils.svm_load('./data/svm.model')

try:
    f = open("./data/test_data.npy")
    f.close()
    test_data=np.load('./data/test_data.npy')
    print('Find')
except IOError:
    print('Not find')      
    utils.save_test_data(test_route)
    test_data=np.load('./data/test_data.npy')
    
try:
    f=open('./data/x.npy')
    f.close()
    f=open('./data/y.npy')
    f.close()
    x=np.load('./data/x.npy')
    y=np.load('./data/y.npy')
    print('X & Y Find')
except:
    print('X & Y not Find')
    x,y=init_xy(test_data,clf)

#x_new=np.linspace(0,1,101)
#f=interpolate.interp1d(x,y,kind='cubic')
#y_new=f(x_new)
#plt.plot(x_new,y_new)
plt.plot(x,y,'*')
plt.plot(x,y)
plt.show()

'''
right=tot=0
for i in range(0,np.size(par)):
    if par[i]==test_label[i]:
        right+=1
print(par)

print('right',right,'tot',tot,'rate',right/tot)
'''