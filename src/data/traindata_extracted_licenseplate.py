import sys
import cv2
import numpy as np
import pandas as pd


class ExtractedLicensePlate:

    def __init__(self, img):
        self.img = img

    # 画像の平滑化とモルフォロジー演算
    def preprocess(self, img):

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gaussian = cv2.GaussianBlur(gray, (5, 5), 0, 0, cv2.BORDER_DEFAULT)
        median = cv2.medianBlur(gaussian, 3)
        sobel = cv2.Sobel(median, cv2.CV_8U, 1, 0, ksize=3)     # ｘ方向に微分
        # canny = cv2.Canny(median, 100, 400)
        ret, binary = cv2.threshold(sobel, 117, 255, cv2.THRESH_BINARY)
        # th3 = cv2.adaptiveThreshold(sobel, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

        element1 = cv2.getStructuringElement(cv2.MORPH_RECT, (6, 1))
        element2 = cv2.getStructuringElement(cv2.MORPH_RECT, (6, 6))

        dilation = cv2.dilate(binary, element2, iterations=1)
        erosion = cv2.erode(dilation, element1, iterations=1)
        dilation2 = cv2.dilate(erosion, element2, iterations=3)

        return dilation2

    # #######################################################################################
    def find_plate_number_region(self, img):
        region = []
        # 画像の輪郭抽出
        contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for i in range(len(contours)):
            cnt = contours[i]
            # 領域が占める面積
            area = cv2.contourArea(cnt)

            # ある程度小さい面積は飛ばす
            if area < 5000:
                continue

            # 回転を考慮した外接矩形（座標と回転情報）
            rect = cv2.minAreaRect(cnt)
            print('rect is: ')
            print(rect)

            # 外接矩形の頂点座標抽出
            box = cv2.boxPoints(rect)
            box = np.int0(box)

            # 縦と横の長さ抽出
            height = abs(box[0][1] - box[2][1])
            width = abs(box[0][0] - box[2][0])
            # 比率
            ratio = float(width) / float(height)
            print(ratio)

            if ratio > 2 or ratio < 0.6:
                continue
            region.append(box)

        return region

    # #######################################################################################
    def detect(self, img):
        # 画像前処理読み込み
        dilation = self.preprocess(self.img)

        # 特徴抽出
        region = self.find_plate_number_region(dilation)

        # 特長点座標の抽出
        for box in region:
            img_cop = img.copy()
            cv2.drawContours(img_cop, [box], -1, (0, 255, 0), 2)
        ys = [box[0, 1], box[1, 1], box[2, 1], box[3, 1]]
        xs = [box[0, 0], box[1, 0], box[2, 0], box[3, 0]]
        ys_sorted_index = np.argsort(ys)
        xs_sorted_index = np.argsort(xs)

        x1 = box[xs_sorted_index[0], 0]
        x2 = box[xs_sorted_index[3], 0]

        y1 = box[ys_sorted_index[0], 1]
        y2 = box[ys_sorted_index[3], 1]

        img_org2 = img.copy()
        img_plate = img_org2[y1:y2, x1:x2]
        cv2.imshow('number plate', img_plate)
        cv2.imwrite('data/processed/license_plate_localization/number_plate.jpg', img_plate)

        cv2.namedWindow('img', cv2.WINDOW_NORMAL)
        cv2.imshow('img', img)

        # 外接矩形画像の保存
        cv2.imwrite('data/processed/license_plate_localization/contours.png', img)

        cv2.waitKey(0)
        cv2.destroyAllWindows()


licenseplate_imagePath = '../../data/loading/JP_20.jpg'
licenseplate_img = cv2.imread(licenseplate_imagePath)
elp = ExtractedLicensePlate(licenseplate_img)
elp.detect(licenseplate_img)
a = elp.preprocess(licenseplate_img)
cv2.imshow('a', a)
cv2.waitKey(0)