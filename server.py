from library.imagezmq import imagezmq
import cv2
import predict
from configparser import ConfigParser

config = ConfigParser()
config.read('config.ini')

host = config.get('server', 'host')
port = config.get('server', 'port')

# initialize the ImageHub object and setup the socket connnection between client and server
imageHub = imagezmq.ImageHub('tcp://'+host+':'+port)

# start looping over all the frames
while True:
	# receive RPi name and frame from the RPi and acknowledge
	# the receipt

	# receive the frame from the client (i.e. receive the frame from the client.py)
	(key, msg, frame) = imageHub.recv_image()

	# check whether client send exit message or not
	# if client send exit message then save the output in the JSON format so call that function from predict.py
	if 'exit' in msg:
		msg = msg.replace('exit', '')
		msg = ''.join(e for e in msg if e.isalnum())
		msg += '_output'
		print('Client exit: ', msg)
		# function of predict.py file to save the output in the JSON format and to pass the time at which client says exit to server
		# file will saved with the exit time of client
		predict.save_output_json(msg)

	# send the frame for further processing to predict.py file
	output_gesture, output_face, output_object, mask_output = predict.predict_output(frame)

	if key != 123:
		imageHub.send_reply({'Error': 'Invalid Key'})
		break
	else:
		# send the reply to client in the JSON form (i.e. send reply to client.py or client_gui.py)
		imageHub.send_reply({ 'gesture': output_gesture, 'face': output_face, 'object': output_object, 'mask': mask_output})#

cv2.destroyAllWindows()