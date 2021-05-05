# -*- coding: utf-8 -*-
"""
Created on Mon May  3 14:00:05 2021

@author: user
"""

import tensorflow as tf
import numpy as np
#from sklearn.preprocessing import LabelBinarizer
import cv2
import os
import threading
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.optimizers import Adam
import math
from tensorflow.keras.callbacks import ModelCheckpoint
from numba import cuda

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
batch_size =32
D_out = 50


class threadsafe_iter:
    """Takes an iterator/generator and makes it thread-safe by
    serializing call to the `next` method of given iterator/generator.
    """

    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def __next__(self):
        with self.lock:
            return self.it.__next__()


def threadsafe_generator(f):
    """A decorator that takes a generator function and makes it thread-safe.
    """
    def g(*a, **kw):
        return threadsafe_iter(f(*a, **kw))
    return g


def MakeOneHot(Y, D_out):
    N = Y.shape[0]
    Z = np.zeros((N, D_out))
    Z[np.arange(N), Y] = 1
    return Z


with open('val.txt', 'r') as f:
    index_val = f.readlines()
    # state=np.random.get_state()
    np.random.shuffle(index_val)
    lable_list_val = [0]*len(index_val)
    k = 0
    Batch_val = []
    for line in index_val:
        img_lable = line.split()
        lable_list_val[k] = int(img_lable[1])
        img_val = load_img(img_lable[0])
        img_val = img_to_array(img_val)
        img_re = cv2.resize(img_val, (64, 64), interpolation=cv2.INTER_AREA)
        Batch_val.append(img_re)
        k += 1
    lable_list_val = np.array(lable_list_val)
    lable_list_val = MakeOneHot(lable_list_val, D_out)
    Batch_val = np.array(Batch_val)



@threadsafe_generator
def batch_iter(path, batch_size=batch_size):
    f = open(path, 'r')
    index = f.readlines()
    while 1:
        # state=np.random.get_state()
        np.random.shuffle(index)
        cnt = 0
        Batch = []
        Y = []
        sample_num = batch_size
        data_num = len(index)
        for line in index:
            img_i = line.split()
            img = load_img(img_i[0])
            img = img_to_array(img)
            #img = cv2.imread(img_i[0])
            img_re = cv2.resize(img, (64, 64), interpolation=cv2.INTER_AREA)
            Batch.append(img_re)
            Y.append(int(img_i[1]))
            cnt += 1
            if cnt == sample_num:
                # print(sample_num)
                data_num = data_num - cnt
                if data_num < batch_size:
                    sample_num = data_num
                cnt = 0
                lable = MakeOneHot(np.array(Y), D_out)
                #kk = np.array(Batch)
                yield (np.array(Batch), lable)
                Batch = []
                Y = []
    f.close()


path = 'train.txt'
batch_iter(path)
f = open(path, 'r')
index = f.readlines()
data_num = len(index)
#sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
print(tf.config.list_physical_devices('GPU'))

class MyModel(tf.keras.Model):

  def __init__(self):
    super(MyModel, self).__init__(name='MyModel', dynamic=False)
    self.conv1 = tf.keras.layers.Conv2D(filters=6, kernel_size=(5, 5), padding='valid',
          input_shape=(64, 64, 3), activation='relu')
    self.pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))
    self.conv2 = tf.keras.layers.Conv2D(filters=16, kernel_size=(5, 5),
          padding='valid', activation='relu')
    self.pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))
    self.flat = tf.keras.layers.Flatten()
    self.dense1 = tf.keras.layers.Dense(1000, activation=tf.nn.relu)
    self.dense2 = tf.keras.layers.Dense(200, activation=tf.nn.relu)
    self.dense3 = tf.keras.layers.Dense(50, activation=tf.nn.softmax)
    #self.dropout = tf.keras.layers.Dropout(0.5)

  def call(self, inputs): 
    # 使用在 `__init__` 建構的 layers 物件，在這裡實作正向傳播
    x = self.conv1(inputs)
    x = self.pool1(x)
    x = self.conv2(x)
    x = self.pool1(x)
    x = self.flat(x)
    x = self.dense1(x)
    x = self.dense2(x)
    x = self.dense3(x)
    return x
    


model = MyModel()
adam = Adam(lr=0.0001)
model.compile(optimizer=adam, loss='categorical_crossentropy',
              metrics=['accuracy'])
save_dir = './checkpoints'
#model.fit(trainData, trainLabels, batch_size=500, epochs=20, verbose=1, shuffle=True)
checkpointer = ModelCheckpoint(os.path.join(save_dir, 'model_{epoch:03d}.hdf5'),
                               verbose=1, save_weights_only=False)
import time 
a1 = time.time()
history = model.fit(batch_iter(path, batch_size=batch_size),
          steps_per_epoch=math.ceil(data_num/batch_size), epochs=20, validation_data=(Batch_val, lable_list_val),
          max_queue_size=4, verbose=1, workers=2, callbacks=[checkpointer])
a2 = time.time()
a = a2 -a1

hist= history.history
import matplotlib.pyplot as plt
plt.plot(history.history['loss'][0:10],color='r',label='Loss train')
plt.plot(history.history['val_loss'][0:10],color='g',label='Loss validation')
plt.legend() 
plt.show() 

plt.plot(history.history['accuracy'][0:10],color='r',label='Train accuracy')
plt.plot(history.history['val_accuracy'][0:10],color='g',label='Validation accuracy')
plt.legend() 
plt.show()

print("time: "+ str(a/60)+" mins")

