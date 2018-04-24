# Camera-Number-Read
This is a program for reading the number catched by the camera. The recognize algorithm is CNN. The training samples is the MNST which is the writing character.
该工程用于摄像头数字的识别
时      间：2018年4月24日
Python版本：Python 3.6
包含的库文件： CV2,keras,numpy

包含的主要内容点：
1、利用OpenCV2 摄像头的开启和图像的读取
2、神经网络模型的读取
3、图像的基本预处理：①彩色转灰度图像、②图像尺寸变换、③图像二值化、④在图像某区域显示文字
4、Keras神经网络的搭建、训练和预测
5、数字防抖动

最终实现功能：
从摄像头读取图片，预测图片中的数字

该程序用法：
1、先运行CNN_MNST_tain.py用CNN建立神经网络模型。采用手写数字的官方样本进行训练，并保存整个神经网络。
2、再运行MAIN.py读取神经网络，得到指定区域的读取数字。
