import numpy as np
import cv2
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

training_data=np.array(training_data,dtype='float32')
training_label=np.array(training_label,dtype='int32')

#print(training_data)

#print(np.size(training_data,0),np.size(training_data,1))
#print(np.size(training_label,0),np.size(training_label,1))

print('Starting training')

#clf=utils.svm_train(training_data,training_label)
clf=utils.svm_load('./data/svm.model')

route='test/pos/crop_000007b.png'
test_data=np.array([utils.cal_HOG(route)],dtype='float32')
print(test_data)
test_label=clf.predict(test_data)

print('Finish training!')

utils.svm_save(clf)

print(test_label)