import numpy as np
import cv2

from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import models, layers

import os, random
from glob import glob


class CNN:

    def __init__(self, get_filepath):
        self.get_filepath = get_filepath

    def make_training_data_labels(self):
        print('start making training data')
        print('please wait until finished..........')

        number_filepath = glob(self.get_filepath + '/*/*.png')
        x = []
        t = []
        for path in number_filepath:
            # 正解ラベル付与用にフォルダの最後の数字／文字列を読込む
            last_number = path.split('/')[-2]
            img = cv2.imread(path)

            # 画像データを行列化して数値データとしてリストに格納する。正解ラベルを付与する
            x.append(img)
            t.append(last_number)

            # # 画像確認用　画像確認必要の場合はコメントアウトを消す
            # cv2.imshow('number plate', img)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

        x = np.array(x).astype('float')
        t = np.array(t).astype('float')
        print('Done!')
        print(type(x), len(x), type(t), len(t))

        return x, t

    def make_train_val_data(self, x, t):
        def reset_random(seed=0):
            os.environ['PYTHONHASHSEED'] = '0'
            random.seed(seed)
            np.random.seed(seed)
            tf.random.set_seed(seed)

        reset_random(0)
        x_train, x_val, t_train, t_val = train_test_split(x, t, test_size=0.3, random_state=0)

        # x_train /= 255
        # t_train /= 255

        return x_train, x_val, t_train, t_val

    def generated_model(self, x_train, x_val, t_train, t_val):
        model = models.Sequential()
        model.add(layers.Conv2D(32, (3, 3), input_shape=(76, 76, 3)))
        model.add(layers.Activation('relu'))
        model.add(layers.Conv2D(8, 3))
        model.add(layers.Activation('relu'))
        model.add(layers.MaxPooling2D(pool_size=(2, 2)))

        model.add(layers.BatchNormalization())
        model.add(layers.Flatten())
        model.add(layers.Dense(50))
        model.add(layers.Activation('relu'))
        model.add(layers.Dense(10))
        model.add(layers.Activation('softmax'))

        optimizer = keras.optimizers.SGD(lr=0.05)
        model.compile(optimizer=optimizer,
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        # x_train = x_train/255
        # x_val = x_val/255
        history_ = model.fit(x_train, t_train,
                             batch_size=50,
                             epochs=50,
                             verbose=1,
                             validation_data=(x_val, t_val))

        model.save('../../data/model/numberplate_keras.h5')
        return history_



get_filepath = '../../data/training'
c = CNN(get_filepath)
x, t = c.make_training_data_labels()
print('x[1] size       : ', x[1].shape)
print('t    size       : ', len(t))
x_train, x_val, t_train, t_val = c.make_train_val_data(x, t)
print('x_train[1] size : ', x_train[1].shape)
print('x_val[1] size   : ', x_val[1].shape)
print('t_train[0] size : ', t_train[0].shape)
history_ = c.generated_model(x_train, x_val, t_train, t_val)



