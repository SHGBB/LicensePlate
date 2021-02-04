'''
 日本のナンバープレートのサイズ調整
 切り取ってきたナンバープレートを「360 x 180」に調整する。
 そして、所定のフォルダへ格納することを目的としている。
'''


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from glob import glob
import cv2

# ナンバープレートサイズの統一
def resize_images(img):
    w, h = 360, 180
    img2 = cv2.imread(img)
    img2_ = cv2.resize(img2, (w, h))
    # imgname = img.split('/')[-1]
    # cv2.imwrite(destination_to_save + '/re' + '{}.jpg'.format(imgname), img2_)

    return img2_
#
# destination_to_save = '../../data/processed/plates/'
# imgfilepath = '../../data/law/plate/*'
#
# for imgs in glob(imgfilepath):
#     resize_images(imgs, destination_to_save)

