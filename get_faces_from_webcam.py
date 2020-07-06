#!/usr/bin/env python3
# _*_ coding: utf-8 _*_

"""
 @DateTime: 17/4/2020 9:33
 @Author:   balanceTan
 @File:     get_faces_from_webcam.py
 @Software: PyCharm

"""
import cv2
import time
import numpy as np
import utils.utils  as utils
from net.mtcnn import mtcnn
from configs import *


class Collect_Face_Data:
    def __init__(self):
        # creat mtcnn object
        self.mtcnn_model = mtcnn()

        # set threshold(Pnet,Rnet,Onet)
        self.threshold = [0.5, 0.8, 0.9]

    def collect_face_data(self, image,wk):

        kk = wk
        # get height,width of image
        height, width, _ = np.shape(image)

        # the bgr format convert to rgb
        Image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # face detection
        rectangles = self.mtcnn_model.detectFace(Image_rgb, self.threshold)

        # if not detect face, return fun
        if len(rectangles) == 0:
            return
        # print("rectangles:",rectangles)

        # rectangles convert to suquare
        rectangles = utils.rect2square(np.array(rectangles, dtype=np.int32))
        rectangles[:, 0] = np.clip(rectangles[:, 0], 0, width)
        rectangles[:, 1] = np.clip(rectangles[:, 1], 0, height)
        rectangles[:, 2] = np.clip(rectangles[:, 2], 0, width)
        rectangles[:, 3] = np.clip(rectangles[:, 3], 0, height)

        for rectangle in rectangles:
            landmark = (np.reshape(rectangle[5:15], (5, 2)) - np.array([int(rectangle[0]), int(rectangle[1])])) / (
                    rectangle[3] - rectangle[1]) * 160

            crop_img = Image_rgb[int(rectangle[1]):int(rectangle[3]), int(rectangle[0]):int(rectangle[2])]
            crop_img = cv2.resize(crop_img, (160, 160))

            new_img, _ = utils.Alignment_1(crop_img, landmark)
            _new_img = cv2.cvtColor(new_img,cv2.COLOR_RGB2BGR)

            if kk == ord('s'):
                cv2.imwrite('collect_face_dataset/{}.jpg'.format(time.time()),_new_img)
                print("collect face  successfully!")


        rectangles = rectangles[:, 0:4]
        for rectangle in rectangles:
            (left, top, right, bottom) = rectangle
            cv2.rectangle(image, (left, top), (right, bottom), (0, 0, 0), 2)



if __name__ == '__main__':

    employ_face = Collect_Face_Data()

    # init camera params
    camera_num, _, fps, width, height = init_camera_params()

    # open camera
    cap = cv2.VideoCapture(int(camera_num))

    # set camera params
    cap.set(5, int(fps))
    cap.set(3, int(width))
    cap.set(4, int(height))

    font = cv2.FONT_HERSHEY_COMPLEX
    while cap.isOpened():
        wk = cv2.waitKey(10)
        ret, frame = cap.read()
        if ret:
            employ_face.collect_face_data(frame,wk)
            cv2.putText(frame, "S: Save current faces", (20, 50), font, 0.8, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(frame, "Q: Quit", (20, 80), font, 0.8, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.imshow("video", frame)

            if wk == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

