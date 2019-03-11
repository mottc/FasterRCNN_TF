import os
import random

import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Model

from configs import *
from data_loaders.data import get_train_data
from model.network import Network


def train():
    fasterRCNN = Network()
    fasterRCNN.build(is_training=True)
    train_op = tf.train.MomentumOptimizer(learning_rate=0.001, momentum=0.9).minimize(fasterRCNN._losses['total_loss'])
    init_op = tf.global_variables_initializer()
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(init_op)

        base_extractor = VGG16(include_top=False)
        extractor = Model(inputs=base_extractor.input, outputs=base_extractor.get_layer('block5_conv3').output)
        train_img_names = os.listdir(TRAIN_IMG_DATA_PATH)
        trained_times = 0

        for epoch in range(1, MAX_EPOCH + 1):
            random.shuffle(train_img_names)
            for train_img_name in train_img_names:
                img_data, boxes, img_info = get_train_data(train_img_name)
                features = extractor.predict(img_data, steps=1)
                sess.run(train_op, feed_dict={fasterRCNN.feature_map: features, fasterRCNN.gt_boxes: boxes,
                                              fasterRCNN.image_info: img_info})

                trained_times += 1
                if trained_times % 10 == 0:
                    total_loss = sess.run(fasterRCNN._losses['total_loss'],
                                          feed_dict={fasterRCNN.feature_map: features, fasterRCNN.gt_boxes: boxes,
                                                     fasterRCNN.image_info: img_info})
                    print('epoch:{}, trained_times:{}, loss:{}'.format(epoch, trained_times, total_loss))

            if epoch % 10 == 0:
                save_path = saver.save(sess, os.path.join(CHECKPOINTS_PATH, "model_" + str(epoch) + ".ckpt"))
                print("Model saved in path: %s" % save_path)
        save_path = saver.save(sess, os.path.join(CHECKPOINTS_PATH, "model_final.ckpt"))
        print("Model saved in path: %s" % save_path)


if __name__ == '__main__':
    train()
