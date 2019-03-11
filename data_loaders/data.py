import os
import xml.etree.ElementTree as ET
import cv2
import numpy as np
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing import image
from configs import *


def load_pascal_annotation(xml_path):
    classes = CLASSES
    tree = ET.parse(xml_path)
    size = tree.find('size')
    original_width = int(size.find('width').text)
    original_height = int(size.find('height').text)
    scale = RESIZED_IMAGE_SIZE[0] / original_height
    if original_height / original_width < RESIZED_IMAGE_SIZE[0] / RESIZED_IMAGE_SIZE[1]:
        scale = RESIZED_IMAGE_SIZE[1] / original_width
    # scale * original <=[600,1000]
    objs = tree.findall('object')
    if not USE_DIFFICULT:
        non_diff_objs = [obj for obj in objs if int(obj.find('difficult').text) == 0]
        objs = non_diff_objs
    num_objs = len(objs)
    boxes = np.zeros((num_objs, 5), dtype=np.float32)

    for ix, obj in enumerate(objs):
        bbox = obj.find('bndbox')
        x1 = float(bbox.find('xmin').text)
        y1 = float(bbox.find('ymin').text)
        x2 = float(bbox.find('xmax').text)
        y2 = float(bbox.find('ymax').text)
        cls = classes.index(obj.find('name').text.strip())
        boxes[ix, :] = [x1 * scale, y1 * scale, x2 * scale, y2 * scale, cls]  # FIX:尺寸变化

    return boxes, [scale * original_height, scale * original_width, scale]


def get_train_data(img_name):
    # xml
    xml_name = img_name.split('.')[0].strip() + '.xml'
    boxes, img_info = load_pascal_annotation(os.path.join(LABEL_PATH, xml_name))

    img = image.load_img(os.path.join(TRAIN_IMG_DATA_PATH, img_name), target_size=(int(img_info[0]), int(img_info[1])))
    print(int(img_info[0]), int(img_info[1]))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    img_data = preprocess_input(x)
    return img_data, boxes, img_info


def get_predict_data(img_name):
    CVimg = cv2.imread(os.path.join(PREDICT_IMG_DATA_PATH, img_name))
    scale = RESIZED_IMAGE_SIZE[0] / CVimg.shape[0]
    if CVimg.shape[0] / CVimg.shape[1] < RESIZED_IMAGE_SIZE[0] / RESIZED_IMAGE_SIZE[1]:
        scale = RESIZED_IMAGE_SIZE[1] / CVimg.shape[1]

    img = image.load_img(os.path.join(TRAIN_IMG_DATA_PATH, img_name),
                         target_size=(int(scale * CVimg.shape[0]), int(scale * CVimg.shape[1])))
    print(int(scale * CVimg.shape[0]), int(scale * CVimg.shape[1]))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    img_data = preprocess_input(x)
    return img_data, [scale * CVimg.shape[0], scale * CVimg.shape[1], scale]
