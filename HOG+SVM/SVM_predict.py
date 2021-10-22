import utils
import numpy as np
import cv2

pos=utils.read_test_pos_route()
neg=utils.read_test_neg_route()

#print(len(pos))

test_label=[]
test_route=[]

for i in pos:
    test_label.append(1)
    test_route.append(i)

for i in neg:
    test_label.append(-1)
    test_route.append(i)

np.array(test_label)
#print(test_label)

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

par=utils.check_pos_or_neg(clf,test_data)

right=tot=0

for i in range(0,np.size(par)):
    if par[i]==test_label[i]:
        right+=1
    tot+=1

print(par)

print('right',right,'tot',tot,'rate',right/tot)