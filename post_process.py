import tifffile as tifi
import numpy as np
import cv2
import time
from skimage import measure,segmentation,morphology,color,data,filters
from scipy import ndimage as ndi
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)


def fill_hole(img):
    # Fill the holes caused by the cluster and watershed

    binary_img = np.where(img > 0,1,0).astype(np.uint8)
    kernel = np.ones((5,5),np.uint16)
    close_img = cv2.morphologyEx(binary_img,cv2.MORPH_CLOSE,kernel)
    opening_img = cv2.morphologyEx(close_img, cv2.MORPH_OPEN, kernel)
    fill_hole_img = ndi.morphology.binary_fill_holes(opening_img).astype(np.uint8)
    
    return fill_hole_img


def watershed(cluster_result_img, origin_img):
    
    label = measure.label(cluster_result_img, connectivity=2)
    props = measure.regionprops(label, intensity_image=origin_img)
    watershed_result = np.zeros_like(cluster_result_img, dtype=np.uint8)

    for _, obj in enumerate(props):
        int_img = obj['intensity_image'] # int image
        bbox = obj['bbox'] # bounding box
        cluster_img_bbox = cluster_result_img[bbox[0]:bbox[2],bbox[1]:bbox[3]]
        
        # # 基于梯度的分水岭
        # markers = filters.rank.gradient(int_img, morphology.disk(3)) <5
        # markers = ndi.label(markers)[0]

        # gradient = filters.rank.gradient(int_img, morphology.disk(3)) #计算梯度
        # seg_result = segmentation.watershed(gradient, markers, mask=cluster_img_bbox) 

        # 基于距离的分水岭
        distance = ndi.distance_transform_edt(int_img)
        seg_result = segmentation.watershed(-distance, markers=cluster_img_bbox, mask=cluster_img_bbox, compactness=10,
                                            watershed_line=True)
        seg_result = np.where(seg_result != 0, 255, 0).astype(np.uint8)  #binary image
        watershed_result[bbox[0]: bbox[2], bbox[1]: bbox[3]] += seg_result

    kernel = np.ones((5,5),np.uint16)
    watershed_result = cv2.morphologyEx(watershed_result, cv2.MORPH_OPEN, kernel)

    return watershed_result


def noize_remove(img):
    # open operation in 'fill_hole' will create some small and meaningless areas
    # delete it to remove noize
    
    contours, hierarchy = cv2.findContours(img,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE) 
    contours = np.array(contours)
    effec_contours = list()
    for i in range(len(contours)):
        ext_left = np.min(contours[i][:,:,0])
        ext_right = np.max(contours[i][:,:,0])
        ext_up = np.min(contours[i][:,:,1])
        ext_down = np.max(contours[i][:,:,1])
        if((ext_right-ext_left) > 10 and (ext_down-ext_up) > 10):
            effec_contours.append(contours[i])
            
    # print(len(effec_contours))
    label_img = np.zeros_like(img)
    label_img = cv2.drawContours(label_img,effec_contours,-1,255,-1) 
    
    return label_img


def fit_ellipse(label_img):

    kernel = np.ones((3,3),np.uint16)
    label_img = cv2.erode(label_img,kernel)
    contours, hierarchy = cv2.findContours(label_img,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    for i in range(len(contours)):
        if((len(contours[i])<5) or (len(contours[i])>1000)):
            continue
        retval = cv2.fitEllipse(contours[i])  # 取其中一个轮廓拟合椭圆
        label_img = cv2.ellipse(label_img, retval, 255, thickness=-1) # 在原图画椭圆
    label_img = cv2.dilate(label_img,kernel)

    return label_img