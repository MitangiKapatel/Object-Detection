from flask import Flask, send_from_directory, request
from flask_restplus import Resource, Api
from werkzeug.datastructures import FileStorage
from zipfile import ZipFile
from configparser import ConfigParser
import cv2
import os
from datetime import datetime
import predict
import numpy as np

config = ConfigParser()
config.read('config.ini')
INSTALLED_PATH = config.get('settings', 'path')
host = config.get('api', 'host')
port = config.get('api', 'port')

app = Flask(__name__)
api = Api(app)

upload_parser = api.parser()
upload_parser.add_argument('file', location='files', type=FileStorage, required=True)
upload_parser.add_argument('timestamp', type=int, required=True)


def video_analysis(filename, timestamp):
    '''
    :param filename: video filename which is then used for video analysis
           timestamp: timestamp value used for logging in the json file
    :return: name of processed filename
    '''

    cap = cv2.VideoCapture('output/' + filename)
    processed_filename = filename.replace('output', 'processed')
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    out = cv2.VideoWriter('output/'+processed_filename, fourcc, 20.0, (640, 480))

    while True:
        ret, frame = cap.read()
        image = frame
        if ret:

            # send the frame for further processing to predict.py file
            output_gesture, output_face, output_object, mask_output = predict.predict_output(frame)

            # getting output for the gesture
            for box in output_gesture:
                # putting the bounding box around the detected hand and show its motion and its gesture
                cv2.rectangle(image, (int(output_gesture[box]['xmin']), int(output_gesture[box]['ymin'])),
                              (int(output_gesture[box]['xmax']), int(output_gesture[box]['ymax'])), (0, 255, 0), 2, 1)
                cv2.putText(image, output_gesture[box]['gesture'],
                            (int(output_gesture[box]['xmin']), int(output_gesture[box]['ymin']) - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
                cv2.putText(image, output_gesture[box]['motion'],
                            (int(output_gesture[box]['xmax']) - 40, int(output_gesture[box]['ymin']) - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)

            # getting output for the face
            for box in output_face:
                # putting the bounding box around the face and show its expression
                cv2.rectangle(image, (int(output_face[box]['xmin']), int(output_face[box]['ymin'])),
                              (int(output_face[box]['xmax']), int(output_face[box]['ymax'])), (0, 0, 255), 2)
                starty = int(output_face[box]['ymin']) - 10 if int(output_face[box]['ymin']) - 10 > 10 else int(
                    output_face[box]['ymin']) + 10
                cv2.putText(image, output_face[box]['expression'], (int(output_face[box]['xmin']), starty),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

            # getting output for the object
            for box in output_object:
                # putting the bounding box around the object
                cv2.rectangle(image, (int(output_object[box]['xmin']), int(output_object[box]['ymin'])),
                              (int(output_object[box]['xmax']), int(output_object[box]['ymax'])), (255, 0, 0), 2)
                cv2.putText(image, output_object[box]['object'],
                            (int(output_object[box]['xmax']) - 100, int(output_object[box]['ymin']) - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 0, 0), 2)

            # to provide mask on the detected object
            masks = []
            if '0' in mask_output:
                alpha = 0.5
                for i in masks:
                    for c in range(3):
                        image[:, :, c] = np.where(np.array(mask_output[i]['mask']) == 1, image[:, :, c] * (1 - alpha) + alpha * masks[i]['color'][c] * 255, image[:, :, c])

            out.write(image)
        else:
            break

    cap.release()
    out.release()
    predict.save_output_json(processed_filename.replace('.mp4', ''))
    return processed_filename.replace('.mp4', '')

@api.route('/upload')
class Upload(Resource):
    @api.expect(upload_parser)
    def post(self):
        args = upload_parser.parse_args()
        uploaded_file = args['file']  # This is FileStorage instance
        timestamp = args['timestamp']
        current = datetime.now()
        filename = ''.join(e for e in str(current) if e.isalnum())
        filename += '_output.mp4'

        # upload the file to the server
        uploaded_file.save(INSTALLED_PATH + '/output/' +filename)

        # pass the filename to process that video
        processed_filename = video_analysis(filename, timestamp)

        # creating zip file of processed video and json file
        zipObj = ZipFile('output/'+ processed_filename +'.zip', 'w')
        zipObj.write('output/' + processed_filename +'.mp4')
        zipObj.write('output/' + processed_filename +'.json')
        zipObj.close()

        # removing the previous upload and prossed file
        os.remove('output/' + filename)
        os.remove('output/' + processed_filename + '.mp4')
        os.remove('output/' + processed_filename + '.json')

        # return the status and url to download the zip file from the server
        return {'Output': 'file uploaded successfully', 'url': request.host_url + 'zipdownload/'+processed_filename+'.zip'}, 201

@app.route('/zipdownload/<path:filename>')
def zipdownload(filename):
    '''
    :param filename: name of the file to be downloaded
    :return: send the file to the requesting client
    '''
    return send_from_directory(INSTALLED_PATH, 'output/'+filename)

if __name__ == '__main__':
    app.run(host=host, port=port, debug=False, threaded=False)