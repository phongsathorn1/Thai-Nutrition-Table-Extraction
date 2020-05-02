import matplotlib.pyplot as plt
import numpy as np
import cv2

def imshow_pair(image1, image2, cmap1=None, cmap2=None):
    fig, axr = plt.subplots(1, 2)
    axr[0].imshow(image1, cmap=cmap1)
    axr[1].imshow(image2, cmap=cmap2)

    for ax in axr:
        ax.axis('off')

def resizeByPercent(image, scale_percent):
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)

    # resize image
    resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)

    return resized

def crop(image, cordinate_start, cordinate_stop):
    x1, y1 = cordinate_start
    x2, y2 = cordinate_stop
    return image[y1:y2, x1:x2]

if __name__ == "__main__":
    pass
