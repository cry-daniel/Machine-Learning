import numpy as np
from matplotlib import pyplot as plt
import math
import cv2
import os
from tqdm import tqdm

stu=np.loadtxt('./data/students003.txt')

def make_video():
    path = 'C:\\Users\\daniel\\Desktop\\ML_hw4\\res\\'
    fps = 10 #视频每秒的帧数
    size = (640, 480) #需要转为视频的图片的尺寸
    #可以使用cv2.resize()进行修改
    video = cv2.VideoWriter("VideoTest2.avi", cv2.VideoWriter_fourcc('I', '4', '2', '0'), fps, size)

    for i in range(540):
        item = path + str(i) + '.png'
        print(item)
        img = cv2.imread(item)
        video.write(img)

    video.release()
    cv2.destroyAllWindows()

#print(stu)

def Euc(x,y,x_,y_): #欧几里得距离
    return math.sqrt((x-x_)**2+(y-y_)**2)

def clu(data,dis=2.5,minn=3):
    label=np.zeros(len(data))
    core=[]
    x_data=[]
    y_data=[]
    for i in range(len(data)):
        [x,y]=data[i]
        x_data.append(x)
        y_data.append(y)
        cnt=0
        for [x_,y_] in data:
            #print(x,y,x_,y_,Euc(x,y,x_,y_))
            #input()
            if Euc(x,y,x_,y_)<dis:
                cnt+=1
        if cnt>=minn:
            core.append([x,y,i])
    cnt=20
    
    def fin(x,y,num):
        if label[num]==0:
            label[num]=cnt
            for i in range(label.size):
                if label[i]==0:
                    [x_,y_]=data[i]
                    if Euc(x,y,x_,y_)<dis:
                        if [x_,y_,i] in core:
                            fin(x_,y_,i)
                        else:
                            label[i]=cnt
                    
    for item in core:
        [x,y,k]=item
        #plt.plot(x,y,'*')
        if label[k]!=0:
            continue
        cnt+=20
        fin(x,y,k)
    return x_data,y_data,label



def div(stu,maxx=5300):
    pnt=0
    step=0
    data=[]
    while (step<=maxx):
        if stu[pnt][0]==step:
            data.append(stu[pnt][2:4])
        else:
            x,y,label=clu(data)
            #print(label)
            plt.clf()
            plt.scatter(x,y,marker='*',c=label)
            #plt.show()
            plt.savefig('./res/'+str(int(step/10))+'.png')
            data=[]
            data.append(stu[pnt][2:4])
            step+=10
        pnt+=1

if __name__=="__main__":
    div(stu)
    make_video()
            
# README
# 首先将数据文件夹的名字改成 data ，data 文件夹下应该有 Readme.txt 和 students003.txt ~
# 然后创建一个文件夹叫 res ，在当前文件夹下就可以啦，它是用来存每一帧的图片
# 可爱上面的 div(stu) 是用来生成 stu 数据集每一帧的聚类图片
# make_video() 则是将这些聚类图片做成视频，其中 make_video() 第一行的 path 记得改成 path 哦~
# 最后运行的结果应该是生成了一个 .avi 的视频，播放它就行啦