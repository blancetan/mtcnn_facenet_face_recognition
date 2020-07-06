#!/usr/bin/env python3
# _*_ coding: utf-8 _*_

"""
 @DateTime: 12/4/2020 10:30
 @Author:   balanceTan
 @File:    face_recognize.py
 @Software: PyCharm

"""
import cv2
import os
import time
import numpy as np
from net.mtcnn import mtcnn
import utils.utils as utils
from net.inception import InceptionResNetV1
import csv
from collections import Counter
from PIL  import Image,ImageDraw,ImageFont
from logs import log
from configs import *
from multiprocessing import Process

# access to  known_face
known_face_encodings = []
known_face_names = []
known_face_infos = []
temp_names = []

# chinese_font
# font_cn = ImageFont.truetype('font/simsun.ttc', 40, encoding='utf-8')

# Englist_font
font_english = cv2.FONT_HERSHEY_SIMPLEX

class face_rec():
    def __init__(self):
        # create mtcnn object
        self.mtcnn_model = mtcnn()

        # set threshold (Pnet,Rnet and Onet)
        self.threshold = [0.6,0.8,0.9]

        # create facenet object and  load  model
        self.facenet_model = InceptionResNetV1()
        self.facenet_model.load_weights(facenet_model_dir)

        face_list = os.listdir(generate_face_dataset)
        for  face in face_list:
            name = face.split(".")[0]
            img = cv2.imread(os.path.join(generate_face_dataset,face))
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            new_img = np.expand_dims( img_rgb, 0)

            # get 128 dims face_features
            face_encoding = utils.calc_128_vec(self.facenet_model,new_img)

            known_face_encodings.append(face_encoding)
            known_face_names.append(name)

        #   # write into file.csv
        # with open("face_feature.csv","w",newline='') as f:
        #     writer = csv.writer(f)
        #     writer.writerows(known_face_encodings)
        # print('write successful! ')

    def recognize(self,image):
        #-----------------------------------------------#
        #  MTCNN  face_detection
        #-----------------------------------------------#
        height,width,_ = np.shape(image)
        Image_rgb = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

        dectFace_stat_time = cv2.getTickCount()
        # faces detection
        rectangles = self.mtcnn_model.detectFace(Image_rgb, self.threshold)
        dectFace_end_time  = cv2.getTickCount()

        spend_time = ((dectFace_end_time-dectFace_stat_time)*1000)/cv2.getTickFrequency()
        # print("spend_time:",spend_time)

        # if not detect face  return
        if len(rectangles)==0:
            return

        # convert to square
        rectangles = utils.rect2square(np.array(rectangles,dtype=np.int32))
        rectangles[:,0] = np.clip(rectangles[:,0],0,width)
        rectangles[:,1] = np.clip(rectangles[:,1],0,height)
        rectangles[:,2] = np.clip(rectangles[:,2],0,width)
        rectangles[:,3] = np.clip(rectangles[:,3],0,height)
        #-----------------------------------------------#
        #   the faces  encoding
        #-----------------------------------------------#
        face_encodings = []
        for rectangle in rectangles:
            landmark = (np.reshape(rectangle[5:15],(5,2)) - np.array([int(rectangle[0]),int(rectangle[1])]))/(rectangle[3]-rectangle[1])*160

            crop_img = Image_rgb[int(rectangle[1]):int(rectangle[3]), int(rectangle[0]):int(rectangle[2])]
            crop_img = cv2.resize(crop_img,(160,160))

            new_img,_ = utils.Alignment_1(crop_img,landmark)
            new_img = np.expand_dims(new_img,0)

            face_encoding = utils.calc_128_vec(self.facenet_model,new_img)
            face_encodings.append(face_encoding)

        face_names = []
        for face_encoding in face_encodings:

            # get  face compare to  known_face_encodings
            matches = utils.compare_faces(known_face_encodings, face_encoding, tolerance = float(tolerance))
            name = "Unknown"

            # fing the mini_distance face
            face_distances = utils.face_distance(known_face_encodings, face_encoding)

            # get the index of the mini_distance face
            best_match_index = np.argmin(face_distances)


            # according  index  to  get  person
            if matches[best_match_index]:
                name = known_face_names[best_match_index]
            face_names.append(name)
            # print("face_names:",face_names)
            temp_names.append(name)
            # print("name:---->{0},face_distances:---->{1},best_match_index:---->{2}".format(name,face_distances[best_match_index],best_match_index))

            #  logging
            # logger.info("name:-->{0},face_distances:-->{1},best_match_index:-->{2}"\
            #     .format(name,face_distances[best_match_index],best_match_index))

            rectangles = rectangles[:,0:4]


            #  draw rectangle
            for (left, top, right, bottom), name in zip(rectangles, face_names):

                cv2.rectangle(image, (left, top), (right, bottom), (0, 0, 255), 2)
                cv2.putText(image, ".....>>", (left + 20, bottom + 25), font_english, 0.75, (255, 255, 255), 2)
                # cv2.putText(Image, name, (left , bottom - 15), font_englist, 0.75, (255, 255, 255), 2)

            if len(temp_names) == int(control_num):

                result_name = Counter(temp_names).most_common(1)[0][0]
                # print("temp_names:",temp_names)
                # logger.info("name:-->{0},face_distances:-->{1},best_match_index:-->{2}"\
                #     .format(name,face_distances[best_match_index],best_match_index))
                logger.info("result:{}".format(result_name))
                temp_names.clear()
                # if result_name is not Unknown,show suessful,else show fail
                if result_name != "Unknown":
                    # # Display English Characters
                    cv2.putText(image,"success",(left + 20 , bottom + 60), font_english, 1, (255, 255, 255), 4)
                    cv2.imwrite("result/{}.jpg".format(result_name),image)
                    # logger.info("Iamge saved  successfully!")
                    time.sleep(0.1)

                    # Display chinese Texts
                    # img_PIL = Image.fromarray(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
                    # draw = ImageDraw.Draw(img_PIL)
                    # draw.text((left + 20 , bottom + 60),"验证成功",fill=(255,255,255),font=font_cn)
                    # # draw.text((100,100),"验证成功",fill=(255,255,255),font=font_chinese)
                    # image = cv2.cvtColor(np.array(img_PIL), cv2.COLOR_RGB2BGR)
                    # cv2.imshow('Video',image)
                    # time.sleep(1)

                else:
                    # # Display English Characters
                    cv2.putText(image, "fail", (left + 40 , bottom + 60), font_english, 1, (255, 255, 255), 4)
                    time.sleep(0.1)

                    # Display chinese Texts
                    # img_PIL = Image.fromarray(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
                    # draw = ImageDraw.Draw(img_PIL)
                    # draw.text((left + 40, bottom + 60), "验证失败",fill=(255, 255, 255),font=font_cn)
                    # # draw.text((100, 100), "验证失败",fill=(255, 255, 255),font=font_chinese)
                    # image = cv2.cvtColor(np.array(img_PIL), cv2.COLOR_RGB2BGR)
                    # cv2.imshow('Video', image)
                    # time.sleep(1)

            # return cv2.cvtColor(np.array(img_PIL), cv2.COLOR_RGB2BGR)

if __name__ == "__main__":

    # init default_config
    raw_face_dataset,generate_face_dataset,facenet_model_dir,\
    tolerance,control_num = init_dafault_params()

    # init camera params
    camera_num,_,fps,width,height = init_camera_params()

    # creat face_rec  object
    employee_faces = face_rec()

    # open webcam
    cap = cv2.VideoCapture(int(camera_num))

    # set camera params
    cap.set(6,cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
    cap.set(5,int(fps))
    cap.set(3,int(width))
    cap.set(4,int(height))

    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            start_time = cv2.getTickCount()
            employee_faces.recognize(frame)
            end_time = cv2.getTickCount()
            fps =(1/(((end_time - start_time)*1000)/cv2.getTickFrequency()))*1000
            cv2.putText(frame, "FPS: {:.2f}".format(fps), (10, 30), 1, 1.5, (0, 0, 255), 2)
            cv2.putText(frame, "Q: Quit", (10, 65), 1, 1.5, (0, 0, 255), 2)
            cv2.imshow('Video', frame)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

