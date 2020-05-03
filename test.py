import numpy as np
import cv2
import matplotlib.pyplot as plt
import pytesseract

import utils
from text_detection import text_detector

def findRectangle(img):
    # find contour
    contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # find second lagest contour
    max_area, max_cnt = 0, None
    second_area, second_cnt = 0, None

    for ing, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        if area > max_area:
            second_area = max_area
            second_cnt = max_cnt
            max_area = area
            max_cnt = cnt
        elif area > second_area:
            second_area = area
            second_cnt = cnt

    return second_area, second_cnt

def order_points(pts):
    # initialzie a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype="float32")

    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    # return the ordered coordinates
    return rect


def four_point_transform(image, pts):
    # obtain a consistent order of the points and unpack them
    # individually
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    # return the warped image
    return warped

def textLabel(image, pt, padding=None):
    # padding (left, top, right, bottom)
    x1 = min(pt[:, 0])
    x2 = max(pt[:, 0])
    y1 = min(pt[:, 1])
    y2 = max(pt[:, 1])

    if padding is not None:
        x1 -= padding[0]
        y1 += padding[1]
        x2 += padding[2]
        y2 -= padding[3]

    label = utils.crop(image, (x1, y1), (x2, y2))
    return label

# image read
img_c = cv2.imread('images/2020-05-01 13.50.10.jpg')
# img_c = cv2.imread("images/2020-05-01 13.51.05.jpg")
# img_c = cv2.imread('images/2020-05-01 13.50.54.jpg')

img_c = cv2.cvtColor(img_c, cv2.COLOR_BGR2RGB)
img_c = cv2.bilateralFilter(img_c, 9, 75, 75)
img = cv2.cvtColor(img_c, cv2.COLOR_BGR2GRAY)

# adaptive gussian treshold
th3 = cv2.adaptiveThreshold(
    img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

area, cnt = findRectangle(th3)

# cropped image
x, y, w, h = cv2.boundingRect(cnt)
img_cropped = utils.crop(img_c, (x, y), (x+w, y+h))

# picked white label only
img_cropped_hsv = cv2.cvtColor(img_cropped, cv2.COLOR_RGB2HSV)
lower_white = np.array([0, 0, 90])
upper_white = np.array([255, 60, 255])
mask = cv2.inRange(img_cropped_hsv, lower_white, upper_white)

kernel = np.ones((11, 11), np.uint8)
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

# Find 4 corner of label
epsilon = 0.08 * cv2.arcLength(cnt, True)
approx = cv2.approxPolyDP(cnt, epsilon, True)

print(approx)
warped = four_point_transform(img_c, approx.reshape(4, 2))

# Covert to gray scale
warped_gray = cv2.cvtColor(warped, cv2.COLOR_RGB2GRAY)

# Make image more shaper
kernel = np.array([
    [-1, -1, -1],
    [-1, 9, -1],
    [-1, -1, -1]
])
warped_gray = cv2.filter2D(warped_gray, -1, kernel)
warped_gray = cv2.bilateralFilter(warped_gray, 11, 25, 75)

warped_gray = cv2.adaptiveThreshold(
    warped_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 13, 3)
# ret, warped_gray = cv2.threshold(warped_gray, 90, 255, cv2.THRESH_BINARY)

# warped_gray = ~warped_gray
# plt.imshow(warped_gray, cmap='gray')
# plt.show()

warped_rgb = cv2.cvtColor(warped_gray, cv2.COLOR_GRAY2RGB)

# Find text area
result = text_detector.detect(warped_rgb)
labels = []

for i, box in enumerate(result):
    cv2.polylines(warped_rgb, [box[:8].astype(np.int32).reshape((-1, 1, 2))], True, color=(0, 255, 0), thickness=2)

for i, box in enumerate(result):
    pt = box[:8].astype(np.int32).reshape((4, 2))
    padding = (5, 5, 5, 5)
    label = textLabel(warped_rgb, pt)
    labels.append(label)

    # Draw boudaries
    font_scale = 1.2
    font = cv2.FONT_HERSHEY_PLAIN
    rectangle_bgr = (0, 255, 0)
    text = "ID : {}".format(i)

    # text_offset_x = 10
    # text_offset_y = warped_rgb.shape[0] - 25
    (text_width, text_height) = cv2.getTextSize(text, font, fontScale=font_scale, thickness=1)[0]

    # cv2.polylines(warped_rgb, [pt.reshape((-1, 1, 2))], True, color=(0, 255, 0), thickness=2)

    box_coords = ((pt[0][0], pt[0][1]), (pt[0][0] + text_width + 5, pt[0][1] - text_height - 12))
    cv2.rectangle(warped_rgb, box_coords[0], box_coords[1], rectangle_bgr, cv2.FILLED)
    cv2.putText(warped_rgb, text, (pt[0][0], pt[0][1]), font, fontScale=font_scale, color=(0, 0, 0), thickness=2)

    text_ocr = pytesseract.image_to_string(label, lang='tha')
    if text_ocr.replace(' ', '') != '':
        print("ID: %d" %(i))
        print("Result \"%s\"" %(text_ocr))
        print("==========")

plt.imshow(warped_rgb)
plt.show()
    # print(pts)
# for i, box in enumerate(result):
#     # cv2.polylines(warped_rgb, [box[:8].astype(np.int32).reshape((-1, 1, 2))], True, color=(0, 255, 0), thickness=2)
#     print(pytesseract.image_to_string(utils.crop(warped_rgb, ), lang='fra'))

# plt.imshow(warped_rgb)
# plt.show()
