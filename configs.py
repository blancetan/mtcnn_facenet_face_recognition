#!/usr/bin/env python3
#_*_ coding: utf-8 _*_

"""
 @DateTime: 23/4/2020 10:03
 @Author:   balanceTan 
 @File:     configs.py
 @Software: PyCharm
 
"""
from logs import log
import configparser
import cv2

logger = log()
config = configparser.ConfigParser()

result = []

def  set_config():

    config['DEFAULT'] = {
                            'RAW_FACE_DATASET' : 'raw_face_dataset',
                            'GENERATE_FACE_DATASET' : 'generate_face_dataset',
                            'FACENET_MODEL_DIR' : 'model_data/facenet_keras.h5',
                            'tolerance' : '0.45',
                            'control_num' : '10'
                        }
    config['CAMERA_PARAMS'] = {
                                'camera_num': 1,
                                'format' : ('M','J','P','G'),
                                'fps' : 30,
                                'width' : 640,
                                'height' : 480

                             }
    with open('config/config.ini','w') as configFile:
        config.write(configFile)
        logger.info('set paramers successful!')

def init_dafault_params():
    config.read('config/config.ini')

    raw_face_dataset = config['DEFAULT']['RAW_FACE_DATASET']
    generate_face_dataset = config['DEFAULT']['GENERATE_FACE_DATASET']
    facenet_model_dir = config['DEFAULT']['FACENET_MODEL_DIR']
    tolerance = config['DEFAULT']['tolerance']
    control_num = config['DEFAULT']['control_num']

    logger.info("raw_face_dataset:{}".format(raw_face_dataset))
    logger.info("generate_face_dataset:{}".format(generate_face_dataset))
    logger.info("facenet_model_dir:{}".format(facenet_model_dir))
    logger.info("tolerance:{}".format(tolerance))
    logger.info("control_num:{}".format(control_num))

    results = [raw_face_dataset, generate_face_dataset, facenet_model_dir, tolerance, control_num]

    return results


def init_camera_params():

    config.read('config/config.ini')
    # Display  camera  params
    camera_num = config['CAMERA_PARAMS']['camera_num']
    format = config['CAMERA_PARAMS']['format']
    fps = config['CAMERA_PARAMS']['fps']
    width = config['CAMERA_PARAMS']['width']
    height = config['CAMERA_PARAMS']['height']

    logger.info("camera number:{}".format(camera_num))
    logger.info("camera format:{}".format(format))
    logger.info("camera fps:{}".format(fps))
    logger.info("camera width:{}".format(width))
    logger.info("camera height:{}".format(height))


    results = [camera_num,format, fps, width, height]

    return results

if __name__ == '__main__':

    # set_config()
    # init_dafault_params()
    init_camera_params()
