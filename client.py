'''
This file is used by client for video analysis
'''

from library.imagezmq import imagezmq
import socket
import time
from datetime import datetime
import cv2
from configparser import ConfigParser
import numpy as np

config = ConfigParser()
config.read('config.ini')

# key is used for authentication between client and server
key = 123

# socket connection between client and server
sender = imagezmq.ImageSender(key, connect_to="tcp://"+config.get('server', 'host')+":"+config.get('server', 'port'))

# get the host name, initialize the video stream, and allow the
# camera sensor to warmup
rpiName = socket.gethostname()
vs = cv2.VideoCapture(0)

time.sleep(2.0)

while True:
    # read the frame from client camera
    ret,frame = vs.read()

    # frame is fliped to show mirror image in the output
    frame = cv2.flip(frame, 1)

    # send frame to the server (i.e. send frame to server.py file) and return the output json file from the server
    output_json = sender.send_image(rpiName, frame)

    # getting output for the gesture
    output_gesture = output_json['gesture']
    for box in output_gesture:
        # putting the bounding box around the detected hand and show its motion and its gesture
        cv2.rectangle(frame, (int(output_gesture[box]['xmin']), int(output_gesture[box]['ymin'])), (int(output_gesture[box]['xmax']), int(output_gesture[box]['ymax'])), (0, 255, 0), 2, 1)
        cv2.putText(frame, output_gesture[box]['gesture'], (int(output_gesture[box]['xmin']), int(output_gesture[box]['ymin']) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
        cv2.putText(frame, output_gesture[box]['motion'], (int(output_gesture[box]['xmax']) - 40, int(output_gesture[box]['ymin']) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)

    # getting output for the face
    output_face = output_json['face']
    for box in output_face:
        # putting the bounding box around the face and show its expression
        cv2.rectangle(frame, (int(output_face[box]['xmin']), int(output_face[box]['ymin'])), (int(output_face[box]['xmax']), int(output_face[box]['ymax'])), (0, 0, 255), 2)
        starty = int(output_face[box]['ymin']) - 10 if int(output_face[box]['ymin']) - 10 > 10 else int(output_face[box]['ymin']) + 10
        cv2.putText(frame, output_face[box]['expression'], (int(output_face[box]['xmin']), starty), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

    # getting output for the object
    output_object = output_json['object']
    for box in output_object:
        # putting the bounding box around the object
        cv2.rectangle(frame, (int(output_object[box]['xmin']), int(output_object[box]['ymin'])),
                      (int(output_object[box]['xmax']), int(output_object[box]['ymax'])), (255, 0, 0), 2)
        cv2.putText(frame, output_object[box]['object'],
                    (int(output_object[box]['xmax']) - 100, int(output_object[box]['ymin']) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 0, 0), 2)

    # to provide mask on the detected object
    masks = []
    if '0' in output_json['mask']:
        masks = output_json['mask']
        alpha = 0.5
        for i in masks:
            for c in range(3):
                frame[:, :, c] = np.where(np.array(masks[i]['mask']) == 1,
                                               frame[:, :, c] * (1 - alpha) + alpha * masks[i]['color'][c] * 255,
                                               frame[:, :, c])

    cv2.imshow(rpiName, frame)
    # detect any kepresses
    key = cv2.waitKey(1) & 0xFF
    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        current = datetime.now()
        output_json = sender.send_image('exit' + str(current), frame)
        break

vs.release()
cv2.destroyAllWindows()