import os
import xml.etree.ElementTree as ET
import cv2
import numpy as np

from net_config import *


def load_pascal_annotation(xml_path):
    classes = ('__background__', 'aeroplane', 'bicycle', 'bird', 'person')
    tree = ET.parse(xml_path)
    size = tree.find('size')
    width = int(size.find('width').text)
    height = int(size.find('height').text)
    scale = RESIZED_IMAGE_SIZE[0] / height
    if height / width < RESIZED_IMAGE_SIZE[0] / RESIZED_IMAGE_SIZE[1]:
        scale = RESIZED_IMAGE_SIZE[1] / width
    # scale * 原图 <=[600,1000]
    objs = tree.findall('object')
    if not USE_DIFFICULT:
        non_diff_objs = [obj for obj in objs if int(obj.find('difficult').text) == 0]
        objs = non_diff_objs
    num_objs = len(objs)
    boxes = np.zeros((num_objs, 4), dtype=np.float32)
    gt_classes = np.zeros((num_objs, len(classes)), dtype=np.float32)

    for ix, obj in enumerate(objs):
        bbox = obj.find('bndbox')
        # Make pixel indexes 0-based
        x1 = float(bbox.find('xmin').text)
        y1 = float(bbox.find('ymin').text)
        x2 = float(bbox.find('xmax').text)
        y2 = float(bbox.find('ymax').text)
        cls = classes.index(obj.find('name').text.lower().strip())
        boxes[ix, :] = [x1, y1, x2, y2]
        gt_classes[ix, cls] = 1.0

    return boxes, gt_classes, [height, width, scale]


def get_data(img_name):
    # 加载XML
    xml_name = img_name.split('.')[0].strip() + '.xml'
    boxes, gt_classes, img_info = load_pascal_annotation(os.path.join(LABEL_PATH, xml_name))

    # 加载图像
    img = cv2.imread(os.path.join(TRAIN_IMG_DATA_PATH, img_name))
    resize_img = cv2.resize(img, (0, 0), fx=img_info[2], fy=img_info[2])
    img_data = resize_img[np.newaxis, :]
    return img_data, boxes, gt_classes, img_info
