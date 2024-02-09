import os
import cv2

from maskrcnn.config import Config
import maskrcnn.model as modellib

import numpy as np
import random
import colorsys
import json

custom_WEIGHTS_PATH = 'models/object_detection/mask_rcnn_obj_0025.h5'

class DatasetConfig(Config):

    # Give the configuration a recognizable name
    NAME = "obj"    
    
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 5  # Background + other classes

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.7

config = DatasetConfig()


class InferenceConfig(config.__class__):
    # Run detection on one image at a time
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()

ROOT_DIR = os.getcwd()
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

modelmine = modellib.MaskRCNN(mode="inference",model_dir = MODEL_DIR, config = config)

modelmine.load_weights(custom_WEIGHTS_PATH, by_name=True)

class_names= {1:'book',2:'bottle',3:'mouse',4:'pen',5:'phone'}

def random_colors(N, bright=True):
    
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors


def apply_mask(image, mask, color, alpha=0.5):
   
    for c in range(3):
        image[:, :, c] = np.where(mask == 1,
                                  image[:, :, c] *
                                  (1 - alpha) + alpha * color[c] * 255,
                                  image[:, :, c])
    return image

def caption(class_ids,class_names,scores=None,captions=None):    
    if not captions:
        class_id = class_ids
        score = scores if scores is not None else None
        label = class_names[class_id]
        caption = "{} {:.3f}".format(label, score) if score else label
    else:
        caption = captions
    return caption

def prediction_object(image):

    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    output = {}
    mask_output = {}
    
    
    results = modelmine.detect([img], verbose=1)
        
    r = results[0]
    
        
    N = r['rois'].shape[0]
    if not N:
        print("\n*** No instances to display *** \n")
    else:
        assert r['rois'].shape[0] == r['masks'].shape[-1] ==  r['class_ids'].shape[0]
        
    # Generate random colors
    colors = random_colors(N)
        
    #masked_image = img.astype(np.uint32).copy()
    masked_image = img.copy()
    for i in range(N):
        output[i] = {}
        mask_output[i] = {}
        color = colors[i]

        # Bounding box
        if not np.any(r['rois'][i]):
            # Skip this instance. Has no bbox. Likely lost in image cropping.
            continue
        (top, left, bottom, right) = r['rois'][i]   
        
        cv2.rectangle(masked_image, (int(left), int(top)), (int(right), int(bottom)), 
                          (0, 255, 0), 2)
        cv2.putText(masked_image, caption(r['class_ids'][i],class_names,r['scores'][i]), (int(right-100), int(top-5)), cv2.FONT_HERSHEY_SIMPLEX,
                          0.65, (0, 255, 0), 2)
        output[i]['xmin'] = str(int(left))
        output[i]['ymin'] = str(int(top))
        output[i]['xmax'] = str(int(right))
        output[i]['ymax'] = str(int(bottom))
        output[i]['object'] = str(caption(r['class_ids'][i],class_names,r['scores'][i]))
        # Mask
        mask =r['masks'][:, :, i]
        mask = mask.tolist()
        mask_output[i]['mask'] = mask
        mask_output[i]['color'] = color
        
        if True:
            masked_image = apply_mask(masked_image, mask, color)
            
    result=masked_image.astype(np.uint8)
        
    return cv2.cvtColor(result, cv2.COLOR_RGB2BGR), output , mask_output