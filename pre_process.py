import cv2
import numpy as np
import tifffile as tifi
from matplotlib import pyplot as plt

# pre_process
def blur(img):
    KsizeX = 5  
    KsizeY = 5  
    blur_img = cv2.GaussianBlur(img, (KsizeX, KsizeY), 0)
    return blur_img


def otsu(img):
    gray = img.astype(np.uint8)
    retval, dst = cv2.threshold(gray,0,255,cv2.THRESH_OTSU)
    return retval, dst


def int_cor(origin_img):
    
    # 亮度校正，将黯淡区域提亮，使各区域亮度均值近似，以提升聚类效果    
    origin_img_mean = np.mean(origin_img)
    shapes = np.shape(origin_img)
    overlay = 0
    x_step = int(shapes[1]/10)
    y_step = int(shapes[0]/10)
    x_nums = 10
    y_nums = 10

    for x_temp in range(x_nums):
        for y_temp in range(y_nums):
            x_end = min(((x_temp+1)*x_step), shapes[1])
            y_end = min(((y_temp+1)*y_step), shapes[0])

            small_image = origin_img[y_temp*y_step : y_end, x_temp*x_step : x_end]
            small_img_mean = np.mean(small_image)
            ratio = origin_img_mean/small_img_mean
            if(ratio>1):
                small_image_cor = small_image * ratio
                origin_img[y_temp*y_step : y_end, x_temp*x_step : x_end] = small_image_cor

    return origin_img