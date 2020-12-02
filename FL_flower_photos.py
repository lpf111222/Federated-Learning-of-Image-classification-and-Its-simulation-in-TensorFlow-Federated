# 本程序使用联邦学习仿真系统tensorflow_federated，实现多个客户端联合训练一个图像分类器的任务
# 本研究的联邦学习，采用横向学习，FederatedAveraging算法
import numpy as np
import tensorflow as tf
import tensorflow_federated as tff
import collections
import random
from tensorflow import keras
from tensorflow.keras import layers
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 数据集下载地址https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz
# 数据集解压后，和本程序文件放在同一路径下，数据集中3670张图片属于5个类
train_dir = ('flower_photos')
NUM_CLASSES = 5
BATCH_SIZE = 32
IMG_SIZE = (160, 160)
IMG_SHAPE = IMG_SIZE + (3,)

#tensorflow_federated中要求，训练数据统一成OrderedDict有序字典格式，并显示注明"x"和"y"
def map_fn(image_batch,labels_batch):return collections.OrderedDict(x=image_batch,y=labels_batch)
#随机生成100个客户端的数据集，以备在训练时被选择
flower_clients_ids = range(100)
def create_client_dataset_fn(str):
    train_dataset = tf.keras.preprocessing.image_dataset_from_directory(train_dir,
                                                                        shuffle=True,
                                                                        seed=int(str),
                                                                        validation_split=0.9,
                                                                        subset="training",
                                                                        image_size=IMG_SIZE,
                                                                        batch_size=BATCH_SIZE)
    return train_dataset.map(map_fn).prefetch(tf.data.experimental.AUTOTUNE)
flower_clients_dataset = tff.simulation.ClientData.from_clients_and_fn(flower_clients_ids,create_client_dataset_fn)
# 每次联邦学习都随机生成num_clients个客户端的数据，本研究中都是10个
def make_random_federated_data(client_data, num_clients):
    random_clientids = random.sample(client_data.client_ids, num_clients)
    return [
        client_data.create_tf_dataset_for_client(x)
        for x in random_clientids
    ]
# input_spec记录训练数据的格式和形状，作为模型的重要参数
flower_sample = make_random_federated_data(flower_clients_dataset,10)
input_spec = flower_sample[0].element_spec

# 建立模型，此分类器模型参考了tensorflow官网上的图像分类器案例
def create_keras_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.experimental.preprocessing.Rescaling(1. / 255, input_shape=IMG_SHAPE),
        tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(NUM_CLASSES)])
    return model

# 为tff训练过程完善模型的参数，loss为多分类的交叉熵，metrics为多分类的准确率
def model_fn():
  keras_model = create_keras_model()
  return tff.learning.from_keras_model(
      keras_model,
      input_spec=input_spec,
      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
      metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
  )

# 建立联邦学习进程fed_avg
fed_avg = tff.learning.build_federated_averaging_process(
    model_fn,
    client_optimizer_fn=lambda:tf.keras.optimizers.Adam(),
    server_optimizer_fn=lambda:tf.keras.optimizers.Adam(),
    use_experimental_simulation_loop = True #使能多GPU并行学习
)


state = fed_avg.initialize() # 初始化联邦学习进程
NUM_ROUNDS = 200 # 进行200轮联邦学习训练
all_metrics = [] #记录每一轮的acc和loss等metrics信息
# 联邦学习进程： 1、收集每个客户端参数  2、汇总和平均模型参数，并在服务器上生成新的更新模型 3、广播给每个客户端
for round_num in range(1, NUM_ROUNDS):
    print("in FL training!")
    # next表示一轮联邦平均
    state, metrics = fed_avg.next(state, make_random_federated_data(flower_clients_dataset,10)) #如果要随机，迭代器里的数据跟着变化
    print('round {:2d}, metrics={}'.format(round_num, metrics))
    all_metrics.append(metrics) #记录每一轮的acc和loss等metrics信息，方便后面绘图展示结果

# 保存训练好的全局模型
fed_avg_model = create_keras_model()
state.model.assign_weights_to(fed_avg_model) #提取全局模型的参数
fed_avg_model.save('fed_avg_model.h5')  #保存模型

# 保存训练中的metrics参数，以备后续分析
f= open('all_metrics.pkl', 'wb')
pickle.dump(all_metrics, f)
f.close()

# 以200轮轮次为横坐标，每次训练的acc和loss为纵坐标，画图查看训练进展
flower_acc = []
flower_loss = []
for i in all_metrics:
    train = i['train']
    flower_acc.append(train['sparse_categorical_accuracy'])
    flower_loss.append(train['loss'])
plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(range(len(flower_acc)),flower_acc, label='Training Accuracy')
plt.legend(loc='lower right')
plt.title('Training Accuracy')
plt.subplot(1, 2, 2)
plt.plot(range(len(flower_loss)), flower_loss, label='Training Loss')
plt.legend(loc='upper right')
plt.title('Training Loss')
plt.savefig('Training Accuracy and Loss.jpg')