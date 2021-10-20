import utils
import configs
import numpy as np
import cv2

pos=utils.read_pos_route()

tot=0

for route in pos:
    tot+=1
    img=utils.read_img(route)
    #print(np.size(img,0),'x',np.size(img,1),'x',np.size(img,2))
    Img=cv2.filter2D(img,-1,configs.kernel_x)
    Img=cv2.filter2D(Img,-1,configs.kernel_y)
    if tot<4:
        print(np.size(Img,0),'x',np.size(Img,1),'x',np.size(Img,2))
        cv2.imshow('x',Img)
        cv2.waitKey(0)