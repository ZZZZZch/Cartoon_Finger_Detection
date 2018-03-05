# coding:utf-8

import os
import cv2
import numpy as np
from time import sleep
import time
import sys
from darkflow.cli import cliHandler

def ZCHIOU(Reframe, GTframe):
	x1 = float(Reframe[0])
	y1 = float(Reframe[1])
	width1 = float(Reframe[2])-float(Reframe[0])
	height1 = float(Reframe[3])-float(Reframe[1])

	x2 = float(GTframe[0])
	y2 = float(GTframe[1])
	width2 = float(GTframe[2])-float(GTframe[0])
	height2 = float(GTframe[3])-float(GTframe[1])

	endx = max(x1+width1,x2+width2)
	startx = min(x1,x2)
	width = width1+width2-(endx-startx)

	endy = max(y1+height1,y2+height2)
	starty = min(y1,y2)
	height = height1+height2-(endy-starty)
	ratio = 0
	if width >=0 and height >= 0:
		Area = width*height
		Area1 = width1*height1
		Area2 = width2*height2
		ratio = Area*1./(Area1+Area2-Area)

	return np.abs(ratio)


def init_feature():
    detector = cv2.xfeatures2d.SURF_create(300)  # 500 is the threshold Hessian value for the detector.
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=100)  # Or pass empty dictionary
    matcher = cv2.FlannBasedMatcher(index_params, search_params)
    return detector, matcher


def filter_matches(kp1, kp2, matches, ratio=0.75):
    good_matches = [m[0] for m in matches if m[0].distance <= m[1].distance * ratio]

    kp_pairs = [(kp1[m.queryIdx], kp2[m.trainIdx]) for m in good_matches]
    p1 = np.float32([kp[0].pt for kp in kp_pairs])
    p2 = np.float32([kp[1].pt for kp in kp_pairs])
    return p1, p2, kp_pairs


if __name__ == '__main__':
    path_name = './cartoon/pages/'
    detector, matcher = init_feature()
    cap = cv2.imread('./cartoon/sample/7.jpg')

    print('Start..')

    time_b = time.time()

    img_frame = cap

    kp_frame, desc_frame = detector.detectAndCompute(img_frame, None)

    if desc_frame is None:
        print('desc_frame is none')

    max_match = 0
    match_or_not = -1
    max_match_pic = 'None'
    for pic_name in os.listdir(path_name):
        pic_name = path_name + pic_name
        img_pic = cv2.imread(pic_name)
        height, width, _ = img_pic.shape
        kp_pic, desc_pic = detector.detectAndCompute(img_pic, None)
        raw_matches = matcher.knnMatch(desc_pic, trainDescriptors=desc_frame, k=2)
        p1, p2, kp_pairs = filter_matches(kp_pic, kp_frame, raw_matches, 0.5)
        if len(p1) > max_match:
            match_or_not = 1
            max_match = len(p1)
            max_match_pic = pic_name

    page = max_match_pic.split('/')[-1].split('.')[0]
    print("发现页数: ", page)

    if page != 'None':

        page_role_path = './cartoon/role/%s/'%page

        max_match = 3
        max_match_role = 'None'
        role = None
        box = None

        for pic_name in os.listdir(page_role_path):
            pic_name = page_role_path + pic_name
            img_pic = cv2.imread(pic_name)
            height, width, _ = img_pic.shape
            kp_pic, desc_pic = detector.detectAndCompute(img_pic, None)
            raw_matches = matcher.knnMatch(desc_pic, trainDescriptors=desc_frame, k=2)
            p1, p2, kp_pairs = filter_matches(kp_pic, kp_frame, raw_matches, 0.5)
            if len(p1) > max_match:
                max_match_role = pic_name
                role = max_match_role.split('/')[-1].split('.')[0].split('_')[1]
                print("发现角色: ", role)
                print('角色特征点: \n', p2)
                box_x1 = min([each[0] for each in p2])
                box_x2 = max([each[0] for each in p2])
                box_y1 = min([each[1] for each in p2])
                box_y2 = max([each[1] for each in p2])

                box = [box_x1, box_y1, box_x2, box_y2]
                print('Cartoon Box : ', box)

        cliHandler(sys.argv)

        if role and box:
            with open("box_log.txt") as f:
                for line in f:
                    fingerbox = line.strip().strip('[').strip(']').split(',')
                    iou = ZCHIOU(fingerbox, box)
                    if iou > 0:
                        print('手指成功锁定', role)

        os.remove('box_log.txt')
        time_e = time.time()
        print("消耗时间: %.2f s "%(time_e - time_b))
