import numbers as np
import cv2
from matplotlib import pyplot as plt

# image read
img_c= cv2.imread('images/2020-05-01 13.50.10.jpg')
img = cv2.imread('images/2020-05-01 13.50.10.jpg',0)
img = cv2.medianBlur(img,5)

# adaptive gussian treshold
th3 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)

# find contour
contours, hierarchy = cv2.findContours( th3, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


# find second lagest contour
max_area,max_cnt=0,None
second_area,second_cnt=0,None
for ing,cnt in enumerate(contours):
    area=cv2.contourArea(cnt)
    if area > max_area:
        second_area=max_area
        second_cnt=max_cnt
        max_area=area
        max_cnt=cnt
    elif area > second_area:
        second_area = area
        second_cnt = cnt

# crate bounding rectangle
x,y,w,h = cv2.boundingRect(second_cnt)
img_c = cv2.rectangle(img_c,(x,y),(x+w,y+h),(0,255,0),2)

# show image
img_c = cv2.resize(img_c, (960, 540))
cv2.imshow('img',img_c)
cv2.waitKey(0)
cv2.destroyAllWindows()
