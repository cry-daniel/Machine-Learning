import numpy as np
import cv2
import utils
import configs
from matplotlib import pyplot as plt

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
'''
x_axis=np.zeros(3780)
for i in range(3780):
    x_axis[i]=i+1

print(pos_training_label)

plt.plot(x_axis,neg_training_data[1242])
plt.show()

exit()
'''
#将正向负向合并为一个矩阵
'''
training_data=np.vstack((pos_training_data,neg_training_data))
training_label=np.vstack((pos_training_label,neg_training_label))

training_data=np.array(training_data,dtype='float32')
training_label=np.array(training_label,dtype='int32')
'''
pos_training_data = np.insert(pos_training_data, 0, values=1, axis=1)
neg_training_data = np.insert(neg_training_data, 0, values=-1, axis=1)
x = np.vstack((pos_training_data, neg_training_data))
np.random.shuffle(x)
training_label = x[:, 0].astype('int')
training_data = x[:, 1:]


print('Starting training')

'''
clf=utils.svm_train(training_data,training_label)
utils.svm_save(clf)
'''
clf=utils.svm_load('./data/svm.model')

print(clf)

test_data=[]

route='test/pos/crop_000026g.png'
test_data.append(utils.cal_HOG(route)[0,:])
route='test/neg/00001163.png'
test_data.append(utils.cal_HOG(route)[0,:])

test_data=np.array(test_data,dtype='float32')
print(test_data)
test_label=clf.predict(test_data)

print('Finish training!')

print(test_label)