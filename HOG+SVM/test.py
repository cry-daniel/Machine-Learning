import cv2
import numpy as np
import matplotlib as plt
import utils

route='test/neg/00001147.png'

clf=utils.svm_load('./data/svm.model')

test_data=np.array([utils.cal_HOG(route)])

print(test_data)

par=utils.check_pos_or_neg(clf,test_data)

print(par)