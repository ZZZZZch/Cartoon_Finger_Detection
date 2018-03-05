import numpy as np
import math
import cv2
import os
import json
#from scipy.special import expit
#from utils.box import BoundBox, box_iou, prob_compare
#from utils.box import prob_compare2, box_intersection
from ...utils.box import BoundBox
from ...cython_utils.cy_yolo2_findboxes import box_constructor

def ZCHIOU(Reframe,GTframe):
	x1 = Reframe[0]
	y1 = Reframe[1]
	width1 = Reframe[2]-Reframe[0]
	height1 = Reframe[3]-Reframe[1]

	x2 = GTframe[0]
	y2 = GTframe[1]
	width2 = GTframe[2]-GTframe[0]
	height2 = GTframe[3]-GTframe[1]

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


def expit(x):
	return 1. / (1. + np.exp(-x))

def _softmax(x):
	e_x = np.exp(x - np.max(x))
	out = e_x / e_x.sum()
	return out

def findboxes(self, net_out):
	# meta
	meta = self.meta
	boxes = list()
	boxes=box_constructor(meta,net_out)
	return boxes

def postprocess(self, net_out, im, save = True):
	"""
	Takes net output, draw net_out, save to disk
	"""
	boxes = self.findboxes(net_out)
	dic_box = {}
	# meta
	meta = self.meta
	threshold = meta['thresh']
	colors = meta['colors']
	labels = meta['labels']
	if type(im) is not np.ndarray:
		imgcv = cv2.imread(im)
	else: imgcv = im
	h, w, _ = imgcv.shape
	
	resultsForJSON = []
	for b in boxes:
		boxResults = self.process_box(b, h, w, threshold)
		if boxResults is None:
			continue
		left, right, top, bot, mess, max_indx, confidence = boxResults
		if mess in dic_box:
			dic_box[mess].append([left, top, right, bot])
		else:
			dic_box[mess] = [[left, top, right, bot]]
		# print(left, right, top, bot, mess, im)
		thick = int((h + w) // 300)
		if self.FLAGS.json:
			resultsForJSON.append({"label": mess, "confidence": float('%.2f' % confidence), "topleft": {"x": left, "y": top}, "bottomright": {"x": right, "y": bot}})
			continue

		cv2.rectangle(imgcv,
			(left, top), (right, bot),
			colors[max_indx], thick)
		cv2.putText(imgcv, mess, (left, top - 12),
			0, 1e-3 * h, colors[max_indx], thick//3)

	keys = list(dic_box.keys())
	if len(keys) > 0:
		print('dic_box[keys[0]] :', dic_box[keys[0]])
		for each_box_0 in dic_box[keys[0]]:
			with open('box_log.txt', 'a') as lg:
				lg.write(str(each_box_0)+"\n")
	else:
		print('Fake...')




	if not save: return imgcv

	outfolder = os.path.join(self.FLAGS.imgdir, 'out')
	img_name = os.path.join(outfolder, os.path.basename(im))

	if self.FLAGS.json:
		textJSON = json.dumps(resultsForJSON)
		textFile = os.path.splitext(img_name)[0] + ".json"
		with open(textFile, 'w') as f:
			f.write(textJSON)
		return


	cv2.imwrite(img_name, imgcv)
