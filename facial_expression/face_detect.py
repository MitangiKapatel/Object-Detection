import numpy as np
import cv2
from configparser import ConfigParser
import facial_expression_recognition

config = ConfigParser()
config.read('config.ini')

INSTALL_PATH = config.get('settings', 'path')

# loading the trained model
net = cv2.dnn.readNetFromCaffe('models/facial_expression/trained_model/deploy.prototxt.txt', 'models/facial_expression/trained_model/pretrained_model.caffemodel')

def detect_face_and_expression(image):
    '''
    :param image: single frame from the video streaming
    :return: image containing bounding box around face with its accuracy and facial expression of the detected face
    '''

    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0,
                                 (300, 300), (104.0, 177.0, 123.0))

    # pass the blob through the network and obtain the detections and
    # predictions
    net.setInput(blob)
    detections = net.forward()

    expression = facial_expression_recognition.recognition_expression(image)

    # loop over the detections
    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with the
        # prediction
        confidence = detections[0, 0, i, 2]

        # filter out weak detections by ensuring the `confidence` is
        # greater than the minimum confidence
        if confidence < 0.5:
            continue

        # compute the (x, y)-coordinates of the bounding box for the
        # object
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")

        # draw the bounding box of the face along with the associated
        # probability
        text = "{:.2f}%".format(confidence * 100)
        y = startY - 10 if startY - 10 > 10 else startY + 10
        cv2.rectangle(image, (startX, startY), (endX, endY),
                      (0, 0, 255), 2)

        cv2.putText(image, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
        cv2.putText(image, expression, (startX+100, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

    return image