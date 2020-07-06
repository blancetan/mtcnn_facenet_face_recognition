#!/usr/bin/env python3
#_*_ coding: utf-8 _*_

"""
 @DateTime: 17/4/2020 16:06
 @Author:   balanceTan 
 @File:     generate_face_data.py
 @Software: PyCharm
 
"""

import os
import cv2
import numpy as np
from net.mtcnn import mtcnn
import utils.utils as utils
from logs import log

def proc_face_data():
    # create mtcnn object
    mtcnn_model = mtcnn()

    # set threshold (Pnet,Rnet and Onet)
    threshold = [0.5,0.8,0.95]

    face_list = os.listdir(RAW_FACE_DATASET)

    for face in face_list:
        try:
            name = face.split(".")[0]
            img = cv2.imread(os.path.join(RAW_FACE_DATASET,face))
            img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

            # detect face
            rectangles = mtcnn_model.detectFace(img, threshold)

            # convert to square
            rectangles = utils.rect2square(np.array(rectangles))

            # image size (160*160 ) for facenet
            rectangle = rectangles[0]

            # mark landmark
            landmark = (np.reshape(rectangle[5:15],(5,2)) - np.array([int(rectangle[0]),int(rectangle[1])]))/(rectangle[3]-rectangle[1])*160

            crop_img = img[int(rectangle[1]):int(rectangle[3]), int(rectangle[0]):int(rectangle[2])]
            crop_img = cv2.resize(crop_img,(160,160))

            new_img,_ = utils.Alignment_1(crop_img,landmark)
            img_rgb = cv2.cvtColor(new_img,cv2.COLOR_BGR2RGB)
            # cv2.imwrite('./proced_face_dataset/{0}.jpg'.format(name),img_rgb)
            cv2.imwrite(os.path.join("GENERATE_FACE_DATASET","{0}.jpg".format(name)),img_rgb)
            logger.info("face data generated successfully!")

        except Exception as ex:
            logger.error(ex)
            logger.error("absPath of excepted image :{}".format(os.path.join(RAW_FACE_DATASET,face)))
            continue

    logger.info("face datas generated  finished!")

if __name__ == "__main__":
    logger = log()
    faces_data = proc_face_data()


