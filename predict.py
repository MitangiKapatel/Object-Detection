# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 11:52:49 2019
"""
from configparser import ConfigParser
import cv2
import sys
import json
import time

config = ConfigParser()
config.read('config.ini')

INSTALL_PATH = config.get('settings', 'path')
sys.path.insert(0, INSTALL_PATH+'/models/hand_gesture/')
sys.path.insert(1, INSTALL_PATH+'/models/facial_expression/')
sys.path.insert(2, INSTALL_PATH+'/models/object_detection/')
sys.path.insert(3, INSTALL_PATH+'/models/object_detection/Mask_RCNN')

import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)

import hand_gesture_predict
import facial_expression_recognition
import object_detection

output_json = {}
output_json['object'] = []
output_json['faces'] = []
output_json['gesture'] = []
'''
cap = cv2.VideoCapture(0)
while True:
    ret,img = cap.read()
    
    image, output_gesture = hand_gesture_predict.detect_hand(img)
    if output_gesture:
        output_json['gesture'].append(output_gesture)

    #image = face_detect.detect_face_and_expression(image)
    image, output_face = facial_expression_recognition.recognition_expression(image)
    if output_face:
        output_json['faces'].append(output_face)

    cv2.imshow("image",image)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

with open(INSTALL_PATH + '/output/output.json', 'w+') as file:
    json.dump(output_json, file)

cap.release()
cv2.destroyAllWindows()
'''
flag = True
pre = int(time.time())
def predict_output(img, timestamp=-1):
    '''
    :param img: single frame from video streaming
           timestamp: timestamp value used for logging in the json file
    :return: detection output in the form of json from the respective model
    '''
    global flag, pre
    if flag:
        # this if condition will be executed for first time only
        pre = int(time.time())
        flag = False

    now = int(time.time())
    diff = now - pre

    # send the frame for gesture detection to hand_gesture model (call the file hand_gesture_predict inside hand_gesture folder)
    image, output_gesture = hand_gesture_predict.detect_hand(img)

    # send the frame for face detection to face_expression model (call the file facial_expression_recognition inside facial_expression folder)
    image, output_face = facial_expression_recognition.recognition_expression(image)

    output_object, mask_output = {}, {}
    if config.get('default', 'instance_segmentation') == 'true':
        # send the frame for object detection to object_detection model (call the file object_detection inside object_detection folder)
        image, output_object, mask_output = object_detection.prediction_object(image)

    timestamp = timestamp if timestamp != -1 else int(config.get('default', 'timestamp'))
    if diff >= timestamp:
        pre = now
        if output_gesture:
            for i in output_gesture:
                output_gesture[i]['timestamp'] = time.ctime(now)
            output_json['gesture'].append(output_gesture)

        if output_face:
            for i in output_face:
                output_face[i]['timestamp'] = time.ctime(now)
            output_json['faces'].append(output_face)

        if config.get('default', 'instance_segmentation') == 'true':
            if output_object:
                for i in output_object:
                    output_object[i]['timestamp'] = time.ctime(now)
                output_json['object'].append(output_object)

    return output_gesture, output_face, output_object, mask_output

def save_output_json(filename):
    # save the output JSON file
    with open(INSTALL_PATH + '/output/' + filename + '.json', 'w+') as file:
        json.dump(output_json, file)