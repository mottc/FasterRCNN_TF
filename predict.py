import os
import cv2
from configs import *
import numpy as np
import tensorflow as tf

from model.network import Network
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Model
from data_loaders.data import get_predict_data


def bbox_transform_inv(boxes, deltas):
    if boxes.shape[0] == 0:
        return np.zeros((0, deltas.shape[1]), dtype=deltas.dtype)

    boxes = boxes.astype(deltas.dtype, copy=False)
    widths = boxes[:, 2] - boxes[:, 0] + 1.0
    heights = boxes[:, 3] - boxes[:, 1] + 1.0
    ctr_x = boxes[:, 0] + 0.5 * widths
    ctr_y = boxes[:, 1] + 0.5 * heights
    ## 0::4表示先取第一个元素，以后每4个取一个
    dx = deltas[:, 0::4]
    dy = deltas[:, 1::4]
    dw = deltas[:, 2::4]
    dh = deltas[:, 3::4]

    pred_ctr_x = dx * widths[:, np.newaxis] + ctr_x[:, np.newaxis]
    pred_ctr_y = dy * heights[:, np.newaxis] + ctr_y[:, np.newaxis]
    pred_w = np.exp(dw) * widths[:, np.newaxis]
    pred_h = np.exp(dh) * heights[:, np.newaxis]
    ## 预测后的（x1,y1,x2,y2）存入 pred_boxes
    pred_boxes = np.zeros(deltas.shape, dtype=deltas.dtype)
    # x1
    pred_boxes[:, 0::4] = pred_ctr_x - 0.5 * pred_w
    # y1
    pred_boxes[:, 1::4] = pred_ctr_y - 0.5 * pred_h
    # x2
    pred_boxes[:, 2::4] = pred_ctr_x + 0.5 * pred_w
    # y2
    pred_boxes[:, 3::4] = pred_ctr_y + 0.5 * pred_h

    return pred_boxes


def clip_boxes(boxes, im_shape):
    # x1 >= 0
    boxes[:, 0::4] = np.maximum(np.minimum(boxes[:, 0::4], im_shape[1] - 1), 0)
    # y1 >= 0
    boxes[:, 1::4] = np.maximum(np.minimum(boxes[:, 1::4], im_shape[0] - 1), 0)
    # x2 < im_shape[1]
    boxes[:, 2::4] = np.maximum(np.minimum(boxes[:, 2::4], im_shape[1] - 1), 0)
    # y2 < im_shape[0]
    boxes[:, 3::4] = np.maximum(np.minimum(boxes[:, 3::4], im_shape[0] - 1), 0)
    return boxes


def visualization(result_list):
    im = cv2.imread('./experiments/experiment1/data/images/AllowUTurn_contest2015_013.jpg')

    for ab in bo:
        cv2.rectangle(im, (int(ab[1]), int(ab[2])), (int(ab[3]), int(ab[4])), (0, 0, 255), 2)
    cv2.imwrite('./20.jpg', im)


def write_txt_result(result_list):
    file_name = result_list[0][0].split('.')[0]
    txt_file = open(os.path.join(TXT_RESULT_PATH, file_name + '.txt'), 'w')
    for result in result_list:
        for element in result:
            txt_file.write(str(element) + ' ')
        txt_file.write('\n')
    txt_file.close()


def nms(detections, NMS_THRESH):
    # TODO：
    pass


def predict():
    fasterRCNN = Network()
    fasterRCNN.build(is_training=False)
    # TODO:加载已训练的网络参数
    with tf.Session() as sess:
        base_extractor = VGG16(include_top=False)
        extractor = Model(inputs=base_extractor.input, outputs=base_extractor.get_layer('block5_conv3').output)
        predict_img_names = os.listdir(PREDICT_IMG_DATA_PATH)

        for predict_img_name in predict_img_names:
            img_data, img_info = get_predict_data(predict_img_name)
            features = extractor.predict(img_data, steps=1)
            rois, scores, regression_parameter = sess.run(
                [fasterRCNN._predictions["rois"], fasterRCNN._predictions["cls_score"],
                 fasterRCNN._predictions["bbox_pred"]],
                feed_dict={fasterRCNN.feature_map: features,
                           fasterRCNN.image_info: img_info})

            boxes = rois[:, 1:5] / img_info[2]
            scores = np.reshape(scores, [scores.shape[0], -1])
            regression_parameter = np.reshape(regression_parameter, [regression_parameter.shape[0], -1])
            pred_boxes = bbox_transform_inv(boxes, regression_parameter)
            pred_boxes = clip_boxes(pred_boxes, [img_info[0] / img_info[2], img_info[1] / img_info[2]])

            result_list = []
            for class_index, class_name in enumerate(CLASSES[1:]):
                class_index += 1  # 因为跳过了背景类别
                cls_boxes = boxes[:, 4 * class_index:4 * (class_index + 1)]
                cls_scores = scores[:, class_index]
                detections = np.hstack((cls_boxes, cls_scores[:, np.newaxis])).astype(np.float32)
                keep = nms(detections, NMS_THRESH)
                detections = detections[keep, :]

                inds = np.where(detections[:, -1] >= CONF_THRESH)[0]  # 筛选结果
                for i in inds:
                    result_for_a_class = []
                    bbox = detections[i, :4]
                    score = detections[i, -1]
                    result_for_a_class.append(predict_img_name)
                    result_for_a_class.append(class_name)
                    result_for_a_class.append(score)
                    for coordinate in bbox:
                        result_for_a_class.append(coordinate)
                    result_list.append(result_for_a_class)
            if len(result_list) == 0:
                continue

            if TXT_RESULT_WANTED:
                write_txt_result(result_list)

            if IS_VISIBLE:
                visualization(result_list)


if __name__ == '__main__':
    predict()
