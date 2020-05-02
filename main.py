import numpy as np
import cv2
import matplotlib.pyplot as plt
from time import sleep

import utils

def removeNoise(image, filter_size=5):
    output = cv2.medianBlur(image, filter_size)
    return output

def resizeByPercent(image, scale_percent):
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)

    # resize image
    resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)

    return resized

def flood_filling(bw_img):
    image = bw_img.copy()
    ## Create mask
    h, w = image.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)

    result = cv2.floodFill(image, mask, (0, 0), 255)
    return image

def connectComponent(bw_img, connectivity=8):
    nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(bw_img, None, None, None, connectivity, cv2.CV_32S)

    sizes = stats[1:, -1] #get CC_STAT_AREA component
    img2 = np.zeros((labels.shape), np.uint8)

    for i in range(0, nlabels - 1):
        if sizes[i] >= 50:   #filter small dotted regions
            img2[labels == i + 1] = 255

    # output = cv2.bitwise_not(img2)
    output = img2
    return output.astype(np.uint8)

def four_corners_sort(pts):
    """ Sort corners: top-left, bot-left, bot-right, top-right"""
    diff = np.diff(pts, axis=1)
    summ = pts.sum(axis=1)
    return np.array([pts[np.argmin(summ)],
                     pts[np.argmax(diff)],
                     pts[np.argmax(summ)],
                     pts[np.argmin(diff)]])
def contour_offset(cnt, offset):
    """ Offset contour because of 5px border """
    cnt += offset
    cnt[cnt < 0] = 0
    return cnt

def find_page_contours(edges, img):
    """ Finding corner points of page contour """
    # Getting contours  
    im2, contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # image = cv2.drawContours(img, contours, -1, (0,255,0), 3)

    # plt.imshow(image)
    # plt.show()
    
    # Finding biggest rectangle otherwise return original corners
    height = edges.shape[0]
    width = edges.shape[1]
    MIN_COUNTOUR_AREA = height * width * 0.3
    MAX_COUNTOUR_AREA = width * height

    max_area = MIN_COUNTOUR_AREA
    page_contour = np.array([[0, 0],
                            [0, height],
                            [width, height],
                            [width, 0]])
    max_cnt = None

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if max_area < area < MAX_COUNTOUR_AREA:
            max_cnt = cnt
            max_area = area
    print(max_cnt)
    print(max_area)

    # for cnt in contours:
    #     perimeter = cv2.arcLength(cnt, True)
    #     approx = cv2.approxPolyDP(cnt, 0.03 * perimeter, True)

    #     if len(approx) == 4:
    #         print("Approx: \n%s" %(approx))
    #         image = cv2.drawContours(img, cnt, -1, (0,255,0), 3)

    #     # Page has 4 corners and it is convex
    #     if (len(approx) == 4 and cv2.isContourConvex(approx) and max_area < cv2.contourArea(approx) < MAX_COUNTOUR_AREA):
    #         max_area = cv2.contourArea(approx)
    #         page_contour = approx[:, 0]

    # Sort corners and offset them
    # page_contour = four_corners_sort(page_contour)
    page_contour = max_cnt
    image = cv2.drawContours(img, max_cnt, -1, (0,255,0), 3)

    plt.imshow(image)
    plt.show()

    return contour_offset(page_contour, (-5, -5))

if __name__ == "__main__":
    image = cv2.imread("images/2020-05-01 13.50.10.jpg")
    # image = cv2.imread("images/2020-05-01 13.51.05.jpg")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    image_gray = cv2.bilateralFilter(image_gray, 11, 75, 75)

    thresh = cv2.adaptiveThreshold(image_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    page_contour = find_page_contours(thresh, image)
    print(page_contour)

    # thresh = removeNoise(thresh, filter_size=3)

    # kernel = np.ones((5, 5), np.uint8)
    # dilation = cv2.dilate(thresh, kernel, iterations=1)

    # closing = cv2.morphologyEx(dilation, cv2.MORPH_CLOSE, kernel)
    # edges = cv2.Canny(dilation, 50, 130, 3)

    # utils.imshow_pair(closing, edges, 'gray', 'gray')
    # plt.show()

    # lines = cv2.HoughLines(closing, 1, np.pi/180, 100)

    # for rho,theta in lines[0]:
    #     a = np.cos(theta)
    #     b = np.sin(theta)
    #     x0 = a*rho
    #     y0 = b*rho
    #     x1 = int(x0 + 1000*(-b))
    #     y1 = int(y0 + 1000*(a))
    #     x2 = int(x0 - 1000*(-b))
    #     y2 = int(y0 - 1000*(a))

    #     cv2.line(closing,(x1,y1),(x2,y2),(0,0,255), 3)


    # page_contour = find_page_contours(edges, image)
    # print(page_contour)
    # page_contour.dot(image)

    # plt.imshow(closing)
    # plt.show()

    # utils.imshow_pair(image_gray, edges, 'gray', 'gray')
    # plt.show()
