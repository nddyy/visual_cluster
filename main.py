from sklearn.cluster import DBSCAN,OPTICS
import hdbscan
import os
import tifffile as tifi
import numpy as np
import sys
import time
import cv2
import pre_process
import post_process


def pre_deal(img):

    time_pre_start = time.time()
    int_cor_img = pre_process.int_cor(img)
    int_cor_img = pre_process.int_cor(int_cor_img) # 进行二次提亮，可有效缩短dbscan耗时
    pretre_img = pre_process.blur(int_cor_img)
    # tif_save('/home/wts/Documents/visual_cluster/int_doubleCor_result.tif', img)
    time_pre_end = time.time()
    print('pre cost time: ', time_pre_end - time_pre_start)

    return pretre_img

def cluster(img):
    # 去除'0'背景值后进行dbscan聚类
    
    img_shape = img.shape
    nonzero_tuple = np.nonzero(img)
    nonzero_np = np.dstack((nonzero_tuple[0],nonzero_tuple[1]))[0]

    time1 = time.time()
    print('cluster start')

    cluster_label = DBSCAN(eps = 5, min_samples = 50).fit_predict(nonzero_np)

    time2 = time.time()
    print('cluster end')
    print('cluster cost time: ', time2 - time1)

    cluster_label = np.uint16(cluster_label)

    # cluster_label存在-1的label，即为噪音点，全部去掉
    cluster_label += 1
    effec_cluster_idx = np.nonzero(cluster_label)[0]
    cluster_result_img = np.zeros([img_shape[0],img_shape[1]],dtype='uint16')
    for i in range(effec_cluster_idx.shape[0]):
        idx = effec_cluster_idx[i]
        cluster_result_img[nonzero_tuple[0][idx],nonzero_tuple[1][idx]] = cluster_label[idx]
    cluster_result_img = np.uint16(cluster_result_img)

    return cluster_result_img


def post_deal(cluster_result_img, origin_img):

    time_post_start = time.time()

    watershed_img = post_process.watershed(cluster_result_img, origin_img)
    fill_hole_img = post_process.fill_hole(watershed_img)
    label_img = post_process.noize_remove(fill_hole_img)
    label_img = post_process.fit_ellipse(label_img)
    label_img = post_process.fit_ellipse(label_img)
    label_img = post_process.fit_ellipse(label_img)
    kernel = np.ones((9,9),np.uint16)
    label_img = cv2.morphologyEx(label_img, cv2.MORPH_OPEN, kernel)

    time_post_end = time.time()
    print('post deal cost time: ', time_post_end - time_post_start)

    return label_img


def tif_read(img_path):
    img = tifi.imread(img_path).astype(np.uint8)
    return img

def tif_save(img_path, img):
    tifi.imsave(img_path, img)


if __name__ == '__main__':

    visual_img_path = '/media/Data1/0411_ster/T33_CellSeg/T33_analysis_test/visual_33681_28721_10000_15000.tif'
    label_result_path = '/home/wts/Documents/visual_cluster/dbscan_cor_blur_eps5_min50_co_fill_nonoize_ellipse_result.tif'

    time0 = time.time()
    
    origin_img = tif_read(visual_img_path)
    pretre_img = pre_deal(origin_img)
    cluster_result_img = cluster(pretre_img)
    post_deal_img = post_deal(cluster_result_img, origin_img)
    tif_save(label_result_path, post_deal_img)
    
    time3 = time.time()
    print('total cost time: ', time3 - time0)