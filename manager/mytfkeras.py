# -*- coding: utf-8 -*-
"""
Created 2020/06/15 20:03:11

@author: okuyama.takahiro
@author: okuyama.takahiro
@author: okuyama.takahiro
@author: okuyama.takahiro
"""

import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras import datasets, layers, models
from keras.layers import Input, Dense, Dropout, Activation, Flatten
from keras.models import Model

(train_images, train_labels),(test_images, test_labels) = datasets.mnist.load_data()

train_images = train_images.reshape((60000,28,28,1))
test_images = test_images.reshape((10000,28,28,1))

# CNNのモデルを作成する
model = tf.keras.models.Sequential([
  tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(28, 28,1)), #(3,3)のフィルタを64種類使い畳み込みを行う
  tf.keras.layers.MaxPooling2D(2,2),                                            #(2,2)の最大プーリング層.入力画像内の(2,2)の領域で最大の数を出力する
  tf.keras.layers.Conv2D(32, (3,3), activation='relu'),                         #(3,3)のフィルタを32種類使い畳み込みを行う
  tf.keras.layers.MaxPooling2D(2,2),                                            #
  tf.keras.layers.Dropout(0.25),                                                #過学習予防。一定割合のノードを不活性化することで過学習を防いで精度を上げる。
  tf.keras.layers.Flatten(),                                                    #一次元配列に変換する
  tf.keras.layers.Dense(128, activation='relu'),                                #
  tf.keras.layers.Dropout(0.35),                                                #
  tf.keras.layers.Dense(10, activation='softmax')                               #
])

model = models.load_model("/content/mnistraining.h5")

model.compile(optimizer='adam',loss = 'sparse_categorical_crossentropy')

historys = model.fit(train_images,train_labels, batch_size= 1000, epochs= 5)
model.evaluate(test_images, test_labels, verbose=2)
import numpy as np

test_image = test_images[0]
test_label = test_labels[0]
print(test_image.shape)
print(test_images.shape)
y = model.predict(test_image.reshape(1,28,28,1))
print(np.argmax(y))
print(test_label)

w = model.layers[0].get_weights()[0]
print(w.shape)
fig = plt.figure(figsize=(10,10))
for i in range(64):
  plt.subplot(8,8,i+1)
  plt.xticks([])
  plt.yticks([])
  plt.grid(False)
  plt.imshow(w[:,:,0,i].reshape(3,3), cmap = plt.cm.binary)
plt.show()
fig.savefig("img_test.png")


