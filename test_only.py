import os
import random

import cv2
import tensorflow as tf

from data_loaders.data import get_data
from configs import *
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Model
from model.network import Network

fasterRCNN = Network()
fasterRCNN.build(True)
train_op = tf.train.MomentumOptimizer().minimize(fasterRCNN._losses['total_loss'])

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    base_extractor = VGG16(include_top=False)
    extractor = Model(inputs=base_extractor.input, outputs=base_extractor.get_layer('block5_conv3').output)
    img_names = os.listdir(TRAIN_IMG_DATA_PATH)
    for epoch in range(2):  # TODO:MAX_EPOCH
        random.shuffle(img_names)
        for img_name in img_names:
            img_data, boxes, gt_classes, img_info = get_data(img_name)
            features = extractor.predict(img_data, steps=1)
            sess.run(train_op, feed_dict={fasterRCNN.feature_map: features, fasterRCNN.gt_boxes: boxes,
                                          fasterRCNN.image_info: img_info})
            if epoch % 10 == 0:
                print(sess.run(fasterRCNN._losses['total_loss'],
                               feed_dict={fasterRCNN.feature_map: features, fasterRCNN.gt_boxes: boxes,
                                          fasterRCNN.image_info: img_info}))





    img_data, boxes, gt_classes, img_info = get_data(img_names[0])
    features = extractor.predict(img_data, steps=1)
    bo = sess.run(fasterRCNN._predictions["rois"],
                  feed_dict={fasterRCNN.feature_map: features, fasterRCNN.gt_boxes: boxes,
                             fasterRCNN.image_info: img_info})
    im = cv2.imread('D:/GitHub/FasterRCNN_TF/experiments/images/AllowUTurn_contest2015_013.jpg')
    print(len(bo))
    for ab in bo:
        cv2.rectangle(im, (int(ab[2]), int(ab[4])), (int(ab[1]), int(ab[3])), (0, 0, 255), 2)
    cv2.imwrite('./500.jpg', im)
