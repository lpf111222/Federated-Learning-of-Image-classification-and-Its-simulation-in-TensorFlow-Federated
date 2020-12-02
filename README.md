基于联邦学习的图像分类器及其在TensorFlow Federated中的仿真
Federated Learning of Image classification and Its simulation in TensorFlow Federated

本实验将一个普通的图片分类模型放入联邦学习仿真框架（TensorFlow Federated）中：多个客户端分布式训练，用联邦平均算法聚合客户端的权重参数形成全局模型，再把全局模型下发给各个客户端。循环迭代此过程，观察训练中的准确率和损失函数，验证横向联邦学习能否训练出一个图片分类模型。

源代码：https://github.com/lpf111222/Federated-Learning-of-Image-classification-and-Its-simulation-in-TensorFlow-Federated
数据集：本实验使用了一个鲜花图片数据集，数据集中3670张图片属于5个类（雏菊daisy、蒲公英dandelion、玫瑰roses、向日葵sunflowers、郁金香tulips）
数据集下载地址：https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz

运行说明：
数据集解压后的flower_photos文件夹，和源代码FL_flower_photos.py文件放在同一路径下，用Python解释器执行FL_flower_photos.py即可。
各类软件库版本：Python 3.8、tensorflow-federated 0.17.0、tensorflow 2.3.1、keras 2.4.3
