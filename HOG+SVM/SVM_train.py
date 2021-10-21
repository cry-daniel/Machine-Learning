import numpy as np
import cv2
from numpy.core.fromnumeric import size
import utils
import configs

#加载数据
pos_training_data=np.load('./data/res_pos.npy')
pos_training_label=np.zeros((np.size(pos_training_data,0),1))
for i in range(0,np.size(pos_training_label)):
    pos_training_label[i]=1
neg_training_data=np.load('./data/res_neg.npy')
neg_training_label=np.zeros((np.size(neg_training_data,0),1))
for i in range(0,np.size(neg_training_label)):
    neg_training_label[i]=-1

pos_len=np.size(pos_training_data,0)
neg_len=np.size(neg_training_data,0)

#将正向负向合并为一个矩阵
training_data=np.vstack((pos_training_data,neg_training_data))
training_label=np.vstack((pos_training_label,neg_training_label))

print(np.size(training_data,1))

training_data=np.array(training_data,dtype='float32')
training_label=np.array(training_label,dtype='float32')

svm=utils.svm_config()

print('Starting training')

utils.svm_train(svm,training_data,training_label)

utils.svm_save(svm,'./data/svm.mat')

print('Finish training!')