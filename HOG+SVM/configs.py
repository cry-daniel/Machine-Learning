import numpy as np

pos_route='./INRIAPerson/train/pos.lst'
neg_route='./INRIAPerson/train/neg.lst'

#sobel 算子
#kernel_x=[[-1,-2,-1],[0,0,0],[1,2,1]]
#kernel_y=[[-1,0,1],[-2,0,2],[-1,0,1]]

kernel_x=[[0,0,0],[1,0,-1],[0,0,0]]
kernel_y=[[0,-1,0],[0,0,0],[0,1,0]]

kernel_x=np.array(kernel_x)
kernel_y=np.array(kernel_y)

cell_x=8
cell_y=8