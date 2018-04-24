# -*- coding: utf-8 -*-
"""
Created on Thu Apr 19 22:23:48 2018

@author: GengDx
"""

import cv2
import keras
cap = cv2.VideoCapture(0)   #开启摄像头
index1='0' #第一个数字的字符初始化
index2='0' #第二个字的字符初始化
index3='0' #第三个字的字符初始化
index='???' #三个连用字符的初始化
model=keras.models.load_model('G:/CNN Model/Number_NET.h5') #读取神经网络模型
temp=0      #用来临时存储数字
i=0         #用来防止抖动的计数
number=0    #总数值的整数数字
n=0         #按下S的次数
while True:

    ret,frame = cap.read()      #读取摄像头
    cvt_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)     #彩色转灰度值
    x, y = 400, 100                                         #显示字符的位置
    cv2.putText(img=cvt_frame, text=index+'mL',
                org=(x, y), fontFace=cv2.FONT_HERSHEY_TRIPLEX,
                fontScale=1, color=(255, 255, 255))

    cv2.imshow('CURIE EYE', cvt_frame) 
    """
    图像分割
    """    
    a=cvt_frame
    b1=a[150:210,430:490]               #截取第一个字符的区域
    b2=a[230:290,430:490]               #截取第二个字符的区域
    b3=a[310:370,430:490]               #截取第三个字符的区域
    b4=a[360:400,430:490]               #截取最下方刻度
    cv2.imshow('number1', b1) 
    cv2.imshow('number2', b2)
    cv2.imshow('number3', b3)
    cv2.imshow('number4', b4)
    """
    图像预处理
    """


#   自动阈值二值化    
    TG1=cv2.adaptiveThreshold(b1,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,41,40)
    TG2=cv2.adaptiveThreshold(b2,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,41,40)
    TG3=cv2.adaptiveThreshold(b3,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,41,40)
    TG1 = TG1/255.
    TG2 = TG2/255.
    TG3 = TG3/255.
#   转换成可识别的尺寸和数值
    TG1=1-cv2.resize(TG1,(28,28))
    TG2=1-cv2.resize(TG2,(28,28))
    TG3=1-cv2.resize(TG3,(28,28))
    TG1=TG1.reshape(-1, 28, 28,1).astype('float32')
    TG2=TG2.reshape(-1, 28, 28,1).astype('float32')
    TG3=TG3.reshape(-1, 28, 28,1).astype('float32')
    classes=model.predict(TG1,batch_size=128)
    for r in classes:
        r=r.tolist()#ndarray转为list
        r=[r[0],r[1]]
        num1=r.index(max(r))
        index1=str(num1)
    classes=model.predict(TG2,batch_size=128)
    for r in classes:
        r=r.tolist()#ndarray转为list
        num2=r.index(max(r))
        index2=str(num2)
    classes=model.predict(TG3,batch_size=128)
    for r in classes:
        r=r.tolist()#ndarray转为list
        num3=r.index(max(r))
        index3=str(num3)
    num=num1*100+num2*10+num3
    if number>100:
        index='???'
    else:
        index=str(number)
    if i==10:
        i=0
        if temp==num:
            number=num
        else:
            number=number
        temp=num
                    
    i=i+1
    #延时1ms读取按键值 
    key=cv2.waitKey(1) & 0xFF        
    #键盘q按下 
    if key == ord('q'): 
    #退出while循环，退出程序
        break
    #按键s读取图片
    if key == ord('s'):
        cv2.imwrite('G:/NN picture sample/yiye number/'+str(n)+'test.jpg',frame)
        n=n+1

#python释放摄像头
cap.release() 
#关闭所有窗口
cv2.destroyAllWindows()
