import cv2
import glob

import data.traindata_extracted_numbers as traindata_numbers
import data.traindata_arrangment_images as traindata_360180
import models.prediction_model_cnn as prediction_model_cnn
import primenumber.prime as prime

if __name__ == '__main__':
    # 予測したいナンバープレートの保存先
    imagePath_prediction = '../data/prediction/plate/renp_0020.jpg.jpg'

    # ナンバープレートから抽出した数字の保存先
    imgSavingPath_prediction = '../data/prediction/number/'

    # ナンバープレートから数字を抽出し、所定の場所に格納する
    # img = cv2.imread(imagePath_prediction)
    img = traindata_360180.resize_images(imagePath_prediction)
    # cv2.imshow('a', img) # 画像確認の場合はアンコメント
    # cv2.waitKey(0)
    tngn = traindata_numbers.GetNumberInNumber()
    tngn.smoothed_images(img)
    tngn.find_bignumber(img, imgSavingPath_prediction)

    # 読み込みナンバープレートを表がしたい場合、下記３行をアンコメントをする
    # a = tngn.smoothed_images(img)
    # cv2.imshow('a', a)
    # cv2.waitKey(0)

    # 学習モデルから予測を実行
    modelPath_prediction = '../data/model/numberplate_keras.h5'
    pmc = prediction_model_cnn.PredictionCNN(modelPath_prediction, imgSavingPath_prediction)
    predictions = pmc.predict_given_data()

    # 素数判定
    pum = prime.UsualMath()

    # ナンバープレートに素数判定結果と約数を記載
    message1, message2 = pum.judgment_prim(predictions)
    for i, message in enumerate(pum.judgment_prim(predictions)):
        y = 25 + i * 50
        cv2.putText(img, message, (0, y), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1, 4)


    cv2.imshow('a', img)
    cv2.waitKey(0)
