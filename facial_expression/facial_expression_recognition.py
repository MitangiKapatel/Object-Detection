import cv2
from configparser import ConfigParser

config = ConfigParser()
config.read('config.ini')

INSTALL_PATH = config.get('settings', 'path')

emotions = ["afraid", "angry", "disgusted", "happy", "neutral", "sad", "surprised"]

# Load model
fisher_face_emotion = cv2.face.FisherFaceRecognizer_create()
fisher_face_emotion.read('models/facial_expression/trained_model/emotion_classifier_model.xml')

faceCascade = cv2.CascadeClassifier('models/facial_expression/trained_model/haarcascade_frontalface_alt.xml')

def find_faces(image):
    '''
    :param image: single frame from the video streaming
    :return: cropped image containing only face and coordinates of bounding box of face
    '''
    coordinates = locate_faces(image)
    cropped_faces = [image[y:y + h, x:x + w] for (x, y, w, h) in coordinates]
    normalized_faces = [normalize_face(face) for face in cropped_faces]
    return zip(normalized_faces, coordinates)

def normalize_face(face):
    '''
    :param face: image of only face
    :return: convert the image into GRAY and resize it to 350x350
    '''
    face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    face = cv2.resize(face, (350, 350))

    return face

def locate_faces(image):
    '''
    :param image: single frame from the video streaming
    :return: coordinates that contain the face
    '''
    faces = faceCascade.detectMultiScale(
        image,
        scaleFactor=1.1,
        minNeighbors=15,
        minSize=(70, 70)
    )
    return faces

def recognition_expression(image):
    '''
    :param image: single frame from the video streaming
    :return: expression of the face
    '''
    temp = image

    output = {}

    i = 1
    for normalized_face, (x, y, w, h) in find_faces(temp):
        output[i] = {}
        emotion_prediction, confidence = fisher_face_emotion.predict(normalized_face)

        starty = y - 10 if y - 10 > 10 else y + 10
        cv2.rectangle(image, (x, y), (w + x, h + y), (0, 0, 255), 2)
        cv2.putText(image, emotions[emotion_prediction], (x, starty), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
        output[i]['xmin'] = str(x)
        output[i]['ymin'] = str(y)
        output[i]['xmax'] = str(w + x)
        output[i]['ymax'] = str(h + y)
        output[i]['expression'] = emotions[emotion_prediction]

        i += 1

    return image, output