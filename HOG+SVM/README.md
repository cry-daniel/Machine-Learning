# HOG+SVM
## 实验准备
+   下载INRIAPerson数据集
+   将train_XXXXXX改名为train
+   创建文件夹data、draw、result，其路径详见下树状图
+   .
├── **data**
├── INRIAPerson
│   ├── 70X134H96
│   │   └── Test
│   │       └── pos
│   ├── 96X160H96
│   │   └── Train
│   │       └── pos
│   ├── **draw**
│   ├── test
│   │   ├── neg -> ../Test/neg
│   │   └── pos
│   ├── Test
│   │   ├── annotations
│   │   ├── neg
│   │   └── pos
│   ├── train
│   │   ├── neg -> ../Train/neg
│   │   └── pos
│   └── Train
│       ├── annotations
│       ├── neg
│       └── pos_
└── **result**


## Steps
+   下例如果提示缺少依赖 XXX，`pip install XXX`即可
+   初始化HOG值
    ```bash
    cd HOG+SVM
    python HOG.py
    ```
+   训练SVM
    ```bash
    python SVM_train.py
    ```
    等待出现 finish training 提示

+   在 config.py 中修改test_route的地址，将它的地址改成你的图片的地址
    随后运行命令：
    ```bash
    python Draw_rect.py
    ```
+   若要画计算AUC的曲线在训练 SVM 完成后运行命令：
    ```bash
    python SVM_predict.py
    ```

## 补充
+   更换不同的 SVM 学习模型更改 utils.py 中的 svm_train() 函数，在函数定义下提供了若干可能的学习模型（被注释的那些），但参数不一定优，可以自己多调调试哪个效果好.
+   计算 HOG 时因为样本过多，为了速度采用了调用库的方法，但 utils.py 中封装好了自己实现的计算 HOG 的函数，不想调库可以更改 utils.py 中的 cal_HOG() 与 cal_neg_HOG() 函数.
+   计算 AUC 的时候其实偷了懒，我对每幅图进行缩放后直接判断，但实际上应该用 Draw_rect.py 中类似的方法进行判断每幅图中是否有人，但效果看起来还可以（雾），有时间的童鞋可以自己试一下.