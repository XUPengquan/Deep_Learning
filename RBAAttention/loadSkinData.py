import os
import cv2
from tqdm import tqdm
import numpy as np
from sklearn.model_selection import train_test_split



def loadDataSet():
    img_files = next(os.walk('./skin-lesion-segmentation/Trainx'))[2]
    msk_files = next(os.walk('./skin-lesion-segmentation/Trainy'))[2]

    img_files.sort()
    msk_files.sort()

    print(img_files)  # ['imgx1.jpg', 'imgx10.jpg', 'imgx100.jpg', 'imgx1000.jpg', 'imgx1001.jpg', 'imgx1002.jpg', 'imgx1003.jpg']
    print(msk_files)  # ['imgy1.jpg', 'imgy10.jpg', 'imgy100.jpg', 'imgy1000.jpg', 'imgy1001.jpg', 'imgy1002.jpg', 'imgy1003.jpg']
    print(len(img_files))  # 2000
    print(len(msk_files))  # 2000

    X = []
    Y = []

    for img_fl  in tqdm(img_files):
        # print(img_fl)  # imgx10.jpg
        if (img_fl.split('.')[-1] == 'jpg'):
            # cv2.INTER_CUBIC 4x4像素邻域的双三次插值,cv2.IMREAD_COLOR 默认使用该种标识。加载一张彩色图片，忽视它的透明度
            img = cv2.imread('./skin-lesion-segmentation/Trainx/{}'.format(img_fl), cv2.IMREAD_COLOR)
            resized_img = cv2.resize(img, (256, 192), interpolation=cv2.INTER_CUBIC)  # (w,h)
            X.append(resized_img)  # train data

    for msk_fl in tqdm(msk_files):
        if (msk_fl.split('.')[-1]=='jpg'):
            # cv2.IMREAD_GRAYSCALE : 加载一张灰度图
            msk = cv2.imread('./skin-lesion-segmentation/Trainy/{}'.format(msk_fl), cv2.IMREAD_GRAYSCALE)
            resized_msk = cv2.resize(msk, (256, 192), interpolation=cv2.INTER_CUBIC)
            Y.append(resized_msk)  # GT image

    # print(len(X)) # 2000
    # print(len(Y)) # 2000


    X = np.array(X)
    Y = np.array(Y)

    # print(X.shape)  # (2000, 192, 256, 3)
    # print(Y.shape)  # (2000, 192, 256)

    # 80% used for train, 20% test
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=3)

    # print(Y_train.shape)  # (1600, 192, 256)
    # print(Y_test.shape)   # (400, 192, 256)
    Y_train = Y_train.reshape((Y_train.shape[0],Y_train.shape[1],Y_train.shape[2],1))
    Y_test = Y_test.reshape((Y_test.shape[0],Y_test.shape[1],Y_test.shape[2],1))

    X_train = X_train / 255
    X_test = X_test / 255
    Y_train = Y_train / 255
    Y_test = Y_test / 255

    Y_train = np.round(Y_train,0)
    Y_test = np.round(Y_test,0)

    print(X_train.shape)  # (1600, 192, 256, 3)
    print(Y_train.shape)  # (1600, 192, 256, 1)
    print(X_test.shape)   # (400, 192, 256, 3)
    print(Y_test.shape)   # (400, 192, 256, 1)

    return X_train, Y_train, X_test, Y_test

if __name__ == '__main__':
    loadDataSet()