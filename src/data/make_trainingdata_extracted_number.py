import cv2
import numpy as np


# #######################################################################################
def preprocess(img):
    # 画像の平滑化
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gaussian = cv2.GaussianBlur(gray, (7,7), 3, 0, cv2.BORDER_DEFAULT)
    median = cv2.medianBlur(gaussian, 3)
    canny = cv2.Canny(median, 100, 150)
    ret, binary = cv2.threshold(canny, 100, 255, cv2.THRESH_BINARY)

    # モルフォロジー演算
    element1 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1))
    element2 = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 4))

    dilation = cv2.dilate(binary, element2, iterations=1)
    erosion = cv2.erode(dilation, element1, iterations=1)
    dilation2 = cv2.dilate(erosion, element2, iterations=3)

    return dilation2


# #######################################################################################
def findPlateNumberRegion(img):
    region = []
    # 画像の輪郭抽出
    labels, contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for i in range(len(contours)):
        cnt = contours[i]
        # 領域が占める面積
        area = cv2.contourArea(cnt)

        # ある程度小さい面積は飛ばす
        if area < 500:
            continue
        print(area)
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

        if ratio > 1.8 or ratio < 0.1:
            continue
        region.append(box)

    return region


# #######################################################################################
def detect(img):
    # 画像前処理読み込み
    dilation = preprocess(img)

    # 特徴抽出
    region = findPlateNumberRegion(dilation)
    print('region:', region)

    labels, contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # ナンバープレート領域内の特徴点抽出
    for box in region:
        im_con = img.copy()
        cv2.drawContours(im_con, [box], -1, (0, 255, 0), 2)

        ys = [box[0, 1], box[1, 1], box[2, 1], box[3, 1]]
        xs = [box[0, 0], box[1, 0], box[2, 0], box[3, 0]]
        ys_sorted_index = np.argsort(ys)
        xs_sorted_index = np.argsort(xs)

        x1 = box[xs_sorted_index[0], 0]
        x2 = box[xs_sorted_index[3], 0]

        y1 = box[ys_sorted_index[0], 1]
        y2 = box[ys_sorted_index[3], 1]

        img_org2 = im_con.copy()
        img_plate = img_org2[y1:y2, x1:x2]
        cv2.imshow('number plate', img_plate)
        cv2.imwrite('../../data/law/number/result' + str(box) + '.png', img_plate)

        # cv2.namedWindow('img', cv2.WINDOW_NORMAL)
        # cv2.imshow('img', img)

        cv2.waitKey(0)
        cv2.destroyAllWindows()


imagePath = '../../data/law/plate/np_0020.jpg'
img = cv2.imread(imagePath)
detect(img)
a = preprocess(img)
cv2.imshow('a', a)
cv2.waitKey(0)