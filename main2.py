import cv2
import numpy as np
import matplotlib.pyplot as plt

import utils

if __name__ == "__main__":
    image = cv2.imread("images/2020-05-01 13.50.10.jpg")
    # image = cv2.imread("images/2020-05-01 13.51.05.jpg")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.bilateralFilter(image, 15, 75, 75)
    # image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # image_gray = cv2.equalizeHist(image_gray)

    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    # hsv = cv2.cvtColor(hsv, cv2.COLOR_BGR2HSV)

    # calculating object histogram
    roihist = cv2.calcHist([hsv],[0, 1], None, [180, 256], [0, 180, 0, 256] )
    plt.imshow(roihist)
    plt.show()

    # ret, thresh = cv2.threshold(image_gray, 180, 255, cv2.THRESH_BINARY)

    # utils.imshow_pair(image_gray, thresh, 'gray', 'gray')
    # plt.show()


    # image_gray = np.float32(image_gray)
    # dst = cv2.cornerHarris(image_gray, 5, 7, 0.04)

    # image[dst>0.01*dst.max()]=[0,0,255]

    # plt.imshow(image_gray, cmap='gray')
    # utils.imshow_pair(image, image_gray, 'gray', 'gray')
    # plt.show()
