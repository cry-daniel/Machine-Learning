import utils
import numpy as np
import cv2

route='test/pos/crop_000002a.png'
img=utils.read_img(route)

utils.show_pic(img,img,img,img)

img=utils.cal_HOG(route)

test_data=np.zeros((1,np.size(img)))
for i in range(0,np.size(img)):
    test_data[0][i]=img[i]
test_data=np.array(test_data,dtype='float32')

svm=utils.svm_load('./data/svm.mat')

print(np.size(test_data,1))

par1=svm.predict(test_data)

print(par1)