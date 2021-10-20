import utils
import configs
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm

pos=utils.read_pos_route()

tot=0

for route in tqdm(pos):
    tot+=1
    img=utils.read_img(route)
    x=np.size(img,0)
    y=np.size(img,1)
    #print(x,y)
    Img_x=cv2.filter2D(img,cv2.CV_16S,configs.kernel_x)
    Img_y=cv2.filter2D(img,cv2.CV_16S,configs.kernel_y)
    Img=abs(Img_x)+abs(Img_y)
    '''
    
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
    '''
    arc_max,Img_max=utils.cal_arctan(Img,Img_x,Img_y)
    #print(arc_max)
    res=np.zeros([100,100,9])
    #print(x/configs.cell_x)
    for i in range(0,int(x/configs.cell_x)):
        for j in range(0,int(y/configs.cell_y)):
            res[i][j]=utils.histogram(Img_max,arc_max,i,j)  
    vec=[]
    #print(int(y/(configs.cell_y))-1)
    for i in range(0,int(x/(configs.cell_x))-1):
        for j in range(0,int(y/(configs.cell_y))-1):
            a=utils.NORM(res,i,j)
            #print(len(a))
            #input()
            for k in a:
                vec.append(k)
    #print(np.size(vec))
    np.array(vec)
    np.save('./data/res'+str(tot)+'.npy',vec)
    