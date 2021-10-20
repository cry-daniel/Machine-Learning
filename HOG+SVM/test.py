import utils
import configs
import numpy as np
import cv2
import matplotlib.pyplot as plt

pos=utils.read_pos_route()

tot=0

for route in pos:
    img=utils.read_img(route)
    plt.subplot(2,2,1)
    plt.title('origin')
    plt.imshow(img)
    #print(np.size(img,0),'x',np.size(img,1),'x',np.size(img,2))
    Img_y=cv2.filter2D(img,cv2.CV_16S,configs.kernel_y)
    plt.subplot(2,2,2)
    plt.title('y position')
    plt.imshow(Img_y)
    Img_x=cv2.filter2D(img,cv2.CV_16S,configs.kernel_x)
    plt.subplot(2,2,3)
    plt.title('x position')
    plt.imshow(Img_x)
    Img=abs(Img_x)+abs(Img_y)
    plt.subplot(2,2,4)
    plt.title('x * y')
    plt.imshow(Img)
    plt.show()
    arc_max,Img_max=utils.cal_arctan(Img,Img_x,Img_y)
    print(arc_max)
    res=utils.histogram(Img_max,arc_max,0,0)
    print(res)
    input()
    