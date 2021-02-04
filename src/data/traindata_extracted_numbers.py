import cv2


class GetNumberInNumber:

    # #######################################################################################
    def smoothed_images(self, img):
        # 画像の平滑化
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gaussian = cv2.GaussianBlur(gray, (3, 3), 3, 0, cv2.BORDER_DEFAULT)
        median = cv2.medianBlur(gaussian, 3)
        canny = cv2.Canny(median, 70, 70)
        sobel = cv2.Sobel(median, cv2.CV_8U, 1, 0, ksize=3)     # ｘ方向に微分
        ret, binary = cv2.threshold(canny, 128, 255, cv2.THRESH_BINARY)

        # モルフォロジー演算
        element1 = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
        element2 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

        dilation = cv2.dilate(binary, element2, iterations=1)
        erosion = cv2.erode(dilation, element1, iterations=1)
        dilation2 = cv2.dilate(erosion, element2, iterations=3)

        return dilation2

    # #######################################################################################
    def find_bignumber(self, img, imgSavingPath):
        # 画像前処理読み込み
        dilation = self.smoothed_images(img)
        contours, hierarchy = cv2.findContours(dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        sorted_ctrs = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[0])

        # 読み込み画像サイズ(エリアサイズ)
        img_area = img.shape[0] * img.shape[1]

        c = 1
        for i, ctr in enumerate(sorted_ctrs):
            # Bounding Boxの取得
            x, y, w, h = cv2.boundingRect(ctr)

            # 領域が占める面積
            roi_area = w * h

            # 読み込み画像全体面積と外接矩形面積の割合
            # roi_ratio = roi_area / img_area

            # 外接矩形の比率
            ratio_box = w / h

            # 外接矩形の面積が一定以上小さいものは除外
            if roi_area < 2000:
                # print('clear roi_area')
                continue

            # 外接矩形と画像エリア面積比率から除外条件
            # if roi_ratio > 0.09 or roi_ratio < 0.009:
            #     continue

            # 外接矩形の比率で除外条件
            if ratio_box > 0.75 or ratio_box < 0.09:
                # print('clear_ratio_box')
                continue

            # 外接矩形を表示させたい場合はrectangleを解除すること
            # cv2.rectangle(img, (x, y), (x + w, y + h), (90, 0, 255), 2)

            img2 = img.copy()
            img_plate = img2[y:y+h, x:x+w]

            # 切り取ってきた数字を正方形化を行う
            max_length = max(w, h)
            top = 0
            bottom = 0
            left = 0
            right = 0
            if max_length > w:
                diff_w = max_length - w
                left = diff_w // 2
                right = diff_w - left
            elif max_length > h:
                diff_h = max_length - h
                top = diff_h // 2
                bottom = max_length - top
            else:
                pass

            img_plate = cv2.copyMakeBorder(img_plate, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
            # cv2.imshow('img', img_plate)

            # 正方形に加工後、画像サイズと統一する（76 x 76）
            w, h = 76, 76
            img_plate_ = cv2.resize(img_plate, (w, h))

            # 画像の保存
            cv2.imwrite(imgSavingPath + str(c) + '.png', img_plate_)
            c += 1
            cv2.waitKey(0)
            cv2.destroyAllWindows()

# #######################################################################################
# imagePath_training = '../../data/processed/plates/renp_0020.jpg.jpg' # 訓練データのナンバープレートを読み込みたときに利用
# imgSavingPath_training = '../../data/processed/training_numbers/result_' # 訓練データの数字を保存したいときにここを利用
# imagePath_prediction = '../../data/prediction/plate/renp_0020.jpg.jpg' # 予測データのナンバープレートを使用したいときに利用
# imgSavingPath_prediction = '../../data/prediction/number' # 予測データを保存したときにここを利用
#
# img = cv2.imread(imagePath_prediction)
# g = GetNumberInNumber()
# g.find_bignumber(img, imgSavingPath_prediction)
# a = g.smoothed_images(img)
# cv2.imshow('a', a)
# cv2.waitKey(0)
# cv2.imwrite('../../data/law/number/_result.png', a　# 説明資料用