# Video Analysis Project

this project consist of main 3 type of models. Hand gesture recognition, Facial Expression Recognition, Object Detection with Instance segmentation.
--------------------------------------------------------------------------------------------------------------------

## main files 

main files to run are client.py for client and sever.py for server.


### requirement.txt

contains all the required python libraries to run the project.
pip install -r requirement.txt


### config.ini

Contains all the configuration parameters required for project.

path - path to where project is located
timestamp - no. of frames after write output to json
instance_segmentation - (true/false) to include instance_segmentation or not
host - server ip address
port - server | api port number where we need to run the project (will be same for client and server)


### client.py

connects with server and sends frame from live webcam. after getting result from server draws bounding boxes to the output frame


### client_gui.py

connects with server and sends frame from live webcam. after getting result from server draws bounding boxes to the output frame with json information with tkinter gui library.


### server.py

Receives image frames from client machine and processes with deep learning models and sends output json file (which contains bounding boxes information)


### api.py

api is made for static video processing after running api.py client can upload a video and getback a processed 
video as ZIP file.


### predict.py

this file calls all the methods from models directory.
--------------------------------------------------------------------------------------------------------------------

## models

models directoiry contains 3 types of model.


### hand_gesture

hand_gesture_predict.py contains all the necessary methods to be called by server.py and api.py


### facial_expression

facial_expression.py contains all the necessary methods to be called by server.py and api.py


### object_detection

object_detection.py contains all the necessary methods to be called by server.py and api.py
--------------------------------------------------------------------------------------------------------------------

## libraries

### imagezmq

a library used for passing image frames from client to server.
--------------------------------------------------------------------------------------------------------------------

## output

All the output will be stored in output directory
--------------------------------------------------------------------------------------------------------------------

## other

these files is not used in project yet but will be useful in future.

### retraning

contains model_retrain.py and notebook file for model retraining.

### model_from_json

contains sample json file with attributes from which model will be trianed

model_from_json.py -- this file loads parameter from json for model training, if parameter is not provided in json then it takes default value for parameter.
--------------------------------------------------------------------------------------------------------------------
