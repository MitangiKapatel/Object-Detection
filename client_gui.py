import tkinter
import cv2
import PIL.Image, PIL.ImageTk
from library.imagezmq import imagezmq
import socket
from datetime import datetime
import numpy as np
from configparser import ConfigParser

config = ConfigParser()
config.read('config.ini')

class App:
    def __init__(self, window, window_title, video_source=0):
        self.window = window
        self.window.title(window_title)
        self.window.geometry("1030x480+0+0")
        self.video_source = video_source

        # open video source (by default this will try to open the computer webcam)
        self.vid = cv2.VideoCapture(video_source)
        if not self.vid.isOpened():
            raise ValueError("Unable to open video source", video_source)

        # Get video source width and height
        self.width = self.vid.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT)

        # Create a canvas that can fit the above video source size
        self.canvas = tkinter.Canvas(window, width = self.width, height = self.height)
        self.canvas.place(width=self.width, height=self.height, x=0, y=0)

        # Listbox for data display
        self.listbox = tkinter.Listbox(window, font=('Arial', 10), width=55, height=28)
        self.listbox.place(x=642, y=0)

        self.index = 1

        # key is used for authentication between client and server
        self.key = 123

        # socket connection between client and server
        self.sender = imagezmq.ImageSender(self.key, connect_to="tcp://"+config.get('server', 'host')+":"+config.get('server', 'port'))
        self.rpiName = socket.gethostname()

        # After it is called once, the update method will be automatically called every delay milliseconds
        self.delay = 15
        self.update()

        self.window.mainloop()

    def update(self):
        # Get a frame from the video source
        if self.vid.isOpened():
            cv2.waitKey(10)
            ret, self.frame = self.vid.read()
            if ret:
                # frame is fliped to show mirror image in the output
                self.frame = cv2.flip(self.frame, 1)

                # send frame to the server (i.e. send frame to server.py file) and return the output json file from the server
                output_json = self.sender.send_image(self.rpiName, self.frame)

                # cv2 read the frame in the BGR format and to show the client in RGB it is converted using cv2
                self.frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)

                # getting output for the gesture
                output_gesture = output_json['gesture']
                for box in output_gesture:
                    # putting the bounding box around the detected hand and show its motion and its gesture
                    cv2.rectangle(self.frame, (int(output_gesture[box]['xmin']), int(output_gesture[box]['ymin'])),
                                  (int(output_gesture[box]['xmax']), int(output_gesture[box]['ymax'])), (0, 255, 0), 2,
                                  1)
                    cv2.putText(self.frame, output_gesture[box]['gesture'],
                                (int(output_gesture[box]['xmin']), int(output_gesture[box]['ymin']) - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
                    cv2.putText(self.frame, output_gesture[box]['motion'],
                                (int(output_gesture[box]['xmax']) - 40, int(output_gesture[box]['ymin']) - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
                    self.listbox.insert(self.index, 'Gesture: ' + output_gesture[box]['gesture'] +' | Xmin: ' + str(int(output_gesture[box]['xmin'])) + ' | Xmax: ' + str(int(output_gesture[box]['xmax'])) + ' | Ymin: ' + str(int(output_gesture[box]['ymin'])) + ' | Ymax: ' + str(int(output_gesture[box]['ymax'])))
                    self.listbox.yview(tkinter.END)
                    self.index += 1

                # getting output for the face
                output_face = output_json['face']
                for box in output_face:
                    # putting the bounding box around the face and show its expression
                    cv2.rectangle(self.frame, (int(output_face[box]['xmin']), int(output_face[box]['ymin'])),
                                  (int(output_face[box]['xmax']), int(output_face[box]['ymax'])), (255, 0, 0), 2)
                    starty = int(output_face[box]['ymin']) - 10 if int(output_face[box]['ymin']) - 10 > 10 else int(
                        output_face[box]['ymin']) + 10
                    cv2.putText(self.frame, output_face[box]['expression'], (int(output_face[box]['xmin']), starty),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 0), 2)
                    self.listbox.insert(self.index, 'Expression: ' + output_face[box]['expression'] + ' | Xmin: ' + str(
                        int(output_face[box]['xmin'])) + ' | Xmax: ' + str(
                        int(output_face[box]['xmax'])) + ' | Ymin: ' + str(
                        int(output_face[box]['ymin'])) + ' | Ymax: ' + str(int(output_face[box]['ymax'])))
                    self.listbox.yview(tkinter.END)
                    self.index += 1

                # getting output for the object
                output_object = output_json['object']
                for box in output_object:
                    # putting the bounding box around the object
                    cv2.rectangle(self.frame, (int(output_object[box]['xmin']), int(output_object[box]['ymin'])),
                                  (int(output_object[box]['xmax']), int(output_object[box]['ymax'])), (255, 0, 0), 2)        
                    cv2.putText(self.frame, output_object[box]['object'], (int(output_object[box]['xmax'])-100, int(output_object[box]['ymin'])-5),cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 0, 0), 2)
                    self.listbox.insert(self.index, 'Object: ' + output_object[box]['object'] + ' | Xmin: ' + str(
                        int(output_object[box]['xmin'])) + ' | Xmax: ' + str(
                        int(output_object[box]['xmax'])) + ' | Ymin: ' + str(
                        int(output_object[box]['ymin'])) + ' | Ymax: ' + str(int(output_object[box]['ymax'])))
                    self.listbox.yview(tkinter.END)
                    self.index += 1

                # to provide mask on the detected object
                masks = []
                if '0' in output_json['mask']:
                    masks = output_json['mask']
                    alpha = 0.5
                    for i in masks:
                        for c in range(3):
                            self.frame[:, :, c] = np.where(np.array(masks[i]['mask']) == 1, self.frame[:, :, c] * (1 - alpha) + alpha * masks[i]['color'][c] * 255, self.frame[:, :, c])

                # put the image into canvas to see the output image into tkinter
                self.photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(self.frame))
                self.canvas.create_image(0, 0, image = self.photo, anchor = tkinter.NW)
            self.window.after(self.delay, self.update)

    def __del__(self):
        if self.vid.isOpened():
            current = datetime.now()
            output_json = self.sender.send_image('exit' + str(current), self.frame)
            self.vid.release()

# Create a window and pass it to the Application object
App(tkinter.Tk(), "Project")