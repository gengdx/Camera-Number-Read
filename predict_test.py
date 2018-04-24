# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 13:10:08 2018
图像预处理及CNN识别数字测试程序，用来调试图像的识别，先运行MAIN后读取了分割图片才能处理，改变b1 b2 b3分别为三个数字的图像
采用自动阈值的方法，最后两位为调节参数。
@author: GengDx
"""
import cv2
import keras
Grayimg=cv2.adaptiveThreshold(b3,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,41,40)
TG1 = Grayimg/255.
TG1=1-cv2.resize(TG1,(28,28))
TG2=TG1.reshape(-1, 28, 28,1).astype('float32')
classes=model.predict(TG2,batch_size=128)
cv2.imshow('input image',TG1)
#等待用户操作  
cv2.waitKey(0)  
#释放所有窗口  
cv2.destroyAllWindows()
for r in classes:
    r=r.tolist()#ndarray转为list
    index=r.index(max(r))
    print(index)
