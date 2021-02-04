import numpy as np
import cv2

from tensorflow.keras import models

from glob import glob


class PredictionCNN:

    def __init__(self, get_model_filepath, get_img_filepath):
        self.get_model_filepath = get_model_filepath
        self.get_img_filepath = get_img_filepath

    def predict_given_data(self):
        model = models.load_model(self.get_model_filepath)
        filepath = sorted(glob(self.get_img_filepath + '/*.png'))

        pred = []
        for path in filepath:
            img = cv2.imread(path)

            x = np.array(img).astype('float')
            x = x[np.newaxis]
            predictions = model.predict(x)
            predict_num = predictions[0].argmax()
            pred.append(predict_num)

        return pred


###############################################################################
# get_model_filepath = '../../data/model/numberplate_keras.h5'
# get_img_filepath = '../../data/prediction/number'
#
# pred = PredictionCNN(get_model_filepath, get_img_filepath)
# x = pred.predict_given_data()
# print(x)
