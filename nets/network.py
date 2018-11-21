import tensorflow as tf
import os
import numpy as np
import net_config


class Network(object):
    def __init__(self):
        pass

    def get_data(self):

        def _parse_function(image_name, gt_boxes, image_info):
            image_string = tf.read_file(image_name)
            image_decoded = tf.image.decode_jpeg(image_string)# TODO:数据格式
            image_resized = tf.image.resize_images(image_decoded, net_config.RESIZED_IMAGE_SIZE)# TODO:提取图片尺寸,not here！！
            # FIXME:从gt_boxes_names到gt_boxes，从image_info_names到image_info
            return image_resized, gt_boxes, image_info
        
        img_names = os.listdir('./train')# TODO:将路径提取出来
        image_name_list = []
        gt_boxes_list = []
        image_info_list = []

        for img_name in img_names:
            image_name_list.append('./train/' + img_name)
            gt_boxes_list.append()
            image_info_list.append()
        
        image_name_list = tf.constant(image_name_list)
        gt_boxes_list = tf.constant(gt_boxes_list)
        image_info_list = tf.constant(image_info_list)
        
        dataset = tf.data.Dataset.from_tensor_slices((image_name_list, gt_boxes_list, image_info_list))
        dataset = dataset.map(_parse_function)
        dataset = dataset.shuffle(buffer_size=50000)# TODO:提取buffer_size大小
        dataset = dataset.repeat()
        dataset = dataset.batch(1)
        
        iterator = dataset.make_one_shot_iterator()
        image_data, gt_boxes, image_info = iterator.get_next()
        return image_data, gt_boxes, image_info

    def get_feature_map(self, image_data):
        pass

    def get_anchors(self, image_info_data):
        height_of_feature_map = tf.to_int32(tf.ceil(self.image_info[0] / np.float32(16)))# TODO:这个16是原图到feature map缩小的倍数,提出来？
        width_of_feature_map = tf.to_int32(tf.ceil(self.image_info[1] / np.float32(16)))
        
        

    def region_proposal(self, feature_map, anchors):
        pass

    def roi_pooling(self):
        pass

    def build(self):
        self.image_data, self.gt_boxes, self.image_info = self.get_data()
        
        feature_map = self.get_feature_map(self.image_data)
        self.anchors = self.get_anchors(self.image_info)
        rois = self.region_proposal(feature_map, self.anchors)
        pool = self.roi_pooling()