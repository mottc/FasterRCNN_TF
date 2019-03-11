import tensorflow as tf

from configs import *
from model.anchors_utils import generate_anchors
from model.target_layer import anchor_target_layer, proposal_target_layer


class Network(object):
    def __init__(self):
        self.feature_map = tf.placeholder(dtype=tf.float32, shape=[1, None, None, 512])  # TODO: feature map 通道数
        self.gt_boxes = tf.placeholder(dtype=tf.float32, shape=[None, 5])  # [x1, y1, x2, y2, cls]
        self.image_info = tf.placeholder(dtype=tf.float32, shape=(3,))  # [height, width, scale],resize后的尺寸

        self._predictions = {}
        self._anchor_targets = {}
        self._proposal_targets = {}
        self._losses = {}
        self._num_anchor_types = len(ANCHOR_SCALES) * len(ANCHOR_RATIOS)
        self._num_classes = len(CLASSES)

    def get_anchors(self):
        # TODO:这个16是原图到feature map缩小的倍数,提出来？
        height_of_feature_map = tf.to_int32(tf.ceil(self.image_info[0] / tf.to_float(16)))
        width_of_feature_map = tf.to_int32(tf.ceil(self.image_info[1] / tf.to_float(16)))
        self.anchors, self.num_of_anchors = generate_anchors(height_of_feature_map, width_of_feature_map, 16,
                                                             ANCHOR_SCALES,
                                                             ANCHOR_RATIOS)

    def get_rois(self, rpn_cls_prob, rpn_bbox_pred, output_size):

        def bbox_transform_inv_tf(boxes, deltas):
            # boxes:生成的anchor
            # deltas:包围框偏移量 [1*height*width*anchor_num,4]
            boxes = tf.cast(boxes, deltas.dtype)
            widths = tf.subtract(boxes[:, 2], boxes[:, 0]) + 1.0
            heights = tf.subtract(boxes[:, 3], boxes[:, 1]) + 1.0
            ctr_x = tf.add(boxes[:, 0], widths * 0.5)
            ctr_y = tf.add(boxes[:, 1], heights * 0.5)

            # 获取包围框预测结果[dx,dy,dw,dh](精修？)
            dx = deltas[:, 0]
            dy = deltas[:, 1]
            dw = deltas[:, 2]
            dh = deltas[:, 3]

            # tf.multiply对应元素相乘
            # pre_x = dx * w + ctr_x
            # pre_y = dy * h + ctr_y
            # pre_w = e**dw * w
            # pre_h = e**dh * h
            pred_ctr_x = tf.add(tf.multiply(dx, widths), ctr_x)
            pred_ctr_y = tf.add(tf.multiply(dy, heights), ctr_y)
            pred_w = tf.multiply(tf.exp(dw), widths)
            pred_h = tf.multiply(tf.exp(dh), heights)

            # 将坐标转换为（xmin,ymin,xmax,ymax）格式
            pred_boxes0 = tf.subtract(pred_ctr_x, pred_w * 0.5)
            pred_boxes1 = tf.subtract(pred_ctr_y, pred_h * 0.5)
            pred_boxes2 = tf.add(pred_ctr_x, pred_w * 0.5)
            pred_boxes3 = tf.add(pred_ctr_y, pred_h * 0.5)

            # 叠加结果输出
            return tf.stack([pred_boxes0, pred_boxes1, pred_boxes2, pred_boxes3], axis=1)

        def clip_boxes_tf(boxes, im_info):
            # 按照图像大小裁剪boxes
            b0 = tf.maximum(tf.minimum(boxes[:, 0], im_info[1] - 1), 0)
            b1 = tf.maximum(tf.minimum(boxes[:, 1], im_info[0] - 1), 0)
            b2 = tf.maximum(tf.minimum(boxes[:, 2], im_info[1] - 1), 0)
            b3 = tf.maximum(tf.minimum(boxes[:, 3], im_info[0] - 1), 0)
            return tf.stack([b0, b1, b2, b3], axis=1)

        scores = rpn_cls_prob[:, :, :, self._num_anchor_types:]
        scores = tf.reshape(scores, shape=(-1,))
        rpn_bbox_pred = tf.reshape(rpn_bbox_pred, shape=(-1, 4))

        proposals = bbox_transform_inv_tf(self.anchors, rpn_bbox_pred)
        proposals = clip_boxes_tf(proposals, self.image_info[:2])
        indices = tf.image.non_max_suppression(proposals, scores, max_output_size=output_size, iou_threshold=0.7)
        boxes = tf.gather(proposals, indices)
        boxes = tf.to_float(boxes)
        scores = tf.gather(scores, indices)
        scores = tf.reshape(scores, shape=(-1, 1))
        # TODO:在每个indices前加入batch内索引，由于目前仅支持每个batch一张图像作为输入所以均为0
        batch_inds = tf.zeros((tf.shape(indices)[0], 1), dtype=tf.float32)
        rois = tf.concat([batch_inds, boxes], 1)
        rois.set_shape([None, 5])
        scores.set_shape([None, 1])
        return rois, scores

    def _proposal_target_layer(self, rois, roi_scores, name):
        with tf.variable_scope(name) as scope:
            rois, roi_scores, labels, bbox_targets, bbox_inside_weights, bbox_outside_weights = tf.py_func(
                proposal_target_layer,
                [rois, roi_scores, self.gt_boxes, self._num_classes],
                [tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32],
                name="proposal_target")

            rois.set_shape([128, 5])
            roi_scores.set_shape([128])
            labels.set_shape([128, 1])
            bbox_targets.set_shape([128, self._num_classes * 4])
            bbox_inside_weights.set_shape([128, self._num_classes * 4])
            bbox_outside_weights.set_shape([128, self._num_classes * 4])

            self._proposal_targets['rois'] = rois
            self._proposal_targets['labels'] = tf.to_int32(labels, name="to_int32")
            self._proposal_targets['bbox_targets'] = bbox_targets
            self._proposal_targets['bbox_inside_weights'] = bbox_inside_weights
            self._proposal_targets['bbox_outside_weights'] = bbox_outside_weights

            return rois, roi_scores

    def _anchor_target_layer(self, rpn_class_score, name):
        with tf.variable_scope(name) as scope:
            rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights = tf.py_func(
                anchor_target_layer,
                [rpn_class_score, self.gt_boxes, self.image_info, 16, self.anchors, self._num_anchor_types],
                [tf.float32, tf.float32, tf.float32, tf.float32],
                name="anchor_target")

            rpn_labels.set_shape([1, 1, None, None])
            rpn_bbox_targets.set_shape([1, None, None, self._num_anchor_types * 4])
            rpn_bbox_inside_weights.set_shape([1, None, None, self._num_anchor_types * 4])
            rpn_bbox_outside_weights.set_shape([1, None, None, self._num_anchor_types * 4])

            rpn_labels = tf.to_int32(rpn_labels, name="to_int32")
            self._anchor_targets['rpn_labels'] = rpn_labels
            self._anchor_targets['rpn_bbox_targets'] = rpn_bbox_targets
            self._anchor_targets['rpn_bbox_inside_weights'] = rpn_bbox_inside_weights
            self._anchor_targets['rpn_bbox_outside_weights'] = rpn_bbox_outside_weights

        return rpn_labels

    def region_proposal(self, is_training):
        def _reshape_layer(bottom, num_dim, name):
            input_shape = tf.shape(bottom)  # 1*H*W*18
            with tf.variable_scope(name) as scope:
                to_caffe = tf.transpose(bottom, [0, 3, 1, 2])  # 1*18*H*W
                reshaped = tf.reshape(to_caffe,
                                      tf.concat(axis=0, values=[[1, num_dim, -1], [input_shape[2]]]))  # 1*2*9H*W
                to_tf = tf.transpose(reshaped, [0, 2, 3, 1])  # 1*9H*W*2
                return to_tf

        def _softmax_layer(bottom, name):
            if name.startswith('rpn_cls_prob_reshape'):
                input_shape = tf.shape(bottom)  # 1*9H*W*2
                bottom_reshaped = tf.reshape(bottom, [-1, input_shape[-1]])  # 9HW*2
                reshaped_score = tf.nn.softmax(bottom_reshaped, name=name)
                return tf.reshape(reshaped_score, input_shape)  # 1*9H*W*2
            return tf.nn.softmax(bottom, name=name)

        rpn = tf.layers.conv2d(inputs=self.feature_map, filters=512, kernel_size=[3, 3], padding='SAME',
                               trainable=is_training)
        rpn_class_score = tf.layers.conv2d(rpn, self._num_anchor_types * 2, [1, 1], trainable=is_training)  # 1*H*W*18

        rpn_cls_score_reshape = _reshape_layer(rpn_class_score, 2, 'rpn_cls_score_reshape')
        rpn_cls_prob_reshape = _softmax_layer(rpn_cls_score_reshape, "rpn_cls_prob_reshape")
        rpn_cls_pred = tf.argmax(tf.reshape(rpn_cls_score_reshape, [-1, 2]), axis=1, name="rpn_cls_pred")  # 9hw*2
        rpn_cls_prob = _reshape_layer(rpn_cls_prob_reshape, self._num_anchor_types * 2, "rpn_cls_prob")  # 1*h*w*18

        rpn_bbox_pred = tf.layers.conv2d(rpn, self._num_anchor_types * 4, [1, 1], trainable=is_training)  # 1*h*w*36
        if is_training:  # TODO:test300,train2000。提取
            rois, scores = self.get_rois(rpn_cls_prob, rpn_bbox_pred, 2000)
            rpn_labels = self._anchor_target_layer(rpn_class_score, "anchor")
            with tf.control_dependencies([rpn_labels]):
                rois, _ = self._proposal_target_layer(rois, scores, "rpn_rois")
        else:
            rois, _ = self.get_rois(rpn_cls_prob, rpn_bbox_pred, 300)

        self._predictions["rpn_cls_score"] = rpn_class_score
        self._predictions["rpn_cls_score_reshape"] = rpn_cls_score_reshape
        self._predictions["rpn_cls_prob"] = rpn_cls_prob
        self._predictions["rpn_cls_pred"] = rpn_cls_pred
        self._predictions["rpn_bbox_pred"] = rpn_bbox_pred
        self._predictions["rois"] = rois
        self._rois = rois

    #
    def roi_pooling(self):
        batch_ids = tf.squeeze(tf.slice(self._rois, [0, 0], [-1, 1], name="batch_id"), [1])
        # 获取包围框归一化后的坐标系（坐标与原图比例）
        bottom_shape = tf.shape(self.feature_map)
        height = (tf.to_float(bottom_shape[1]) - 1.) * tf.to_float(16)  # TODO:
        width = (tf.to_float(bottom_shape[2]) - 1.) * tf.to_float(16)
        x1 = tf.slice(self._rois, [0, 1], [-1, 1], name="x1") / width
        y1 = tf.slice(self._rois, [0, 2], [-1, 1], name="y1") / height
        x2 = tf.slice(self._rois, [0, 3], [-1, 1], name="x2") / width
        y2 = tf.slice(self._rois, [0, 4], [-1, 1], name="y2") / height

        bboxes = tf.stop_gradient(tf.concat([y1, x1, y2, x2], axis=1))
        pre_pool_size = 7 * 2
        crops = tf.image.crop_and_resize(self.feature_map, bboxes, tf.to_int32(batch_ids),
                                         [pre_pool_size, pre_pool_size],
                                         name="crops")

        self._pool = tf.layers.max_pooling2d(crops, [2, 2], 2, 'same')

    #
    def head_to_tail(self, is_training):
        pool_flatten = tf.layers.flatten(self._pool)
        fc6 = tf.layers.dense(pool_flatten, 4096)
        if is_training:
            fc6 = tf.layers.dropout(fc6, rate=0.5, training=True)
        self._fc7 = tf.layers.dense(fc6, 4096)
        if is_training:
            self._fc7 = tf.layers.dropout(self._fc7, rate=0.5, training=True)

    #
    def region_classification(self, is_training):

        cls_score = tf.layers.dense(self._fc7, self._num_classes, trainable=is_training)
        cls_prob = tf.nn.softmax(cls_score)
        cls_pred = tf.argmax(cls_score, axis=1)
        bbox_pred = tf.layers.dense(self._fc7, self._num_classes * 4, trainable=is_training)

        self._predictions["cls_score"] = cls_score
        self._predictions["cls_pred"] = cls_pred
        self._predictions["cls_prob"] = cls_prob
        self._predictions["bbox_pred"] = bbox_pred

    #
    def _smooth_l1_loss(self, bbox_pred, bbox_targets, bbox_inside_weights, bbox_outside_weights, sigma=1.0, dim=[1]):
        sigma_2 = sigma ** 2
        box_diff = bbox_pred - bbox_targets
        in_box_diff = bbox_inside_weights * box_diff
        abs_in_box_diff = tf.abs(in_box_diff)
        smoothL1_sign = tf.stop_gradient(tf.to_float(tf.less(abs_in_box_diff, 1. / sigma_2)))
        in_loss_box = tf.pow(in_box_diff, 2) * (sigma_2 / 2.) * smoothL1_sign \
                      + (abs_in_box_diff - (0.5 / sigma_2)) * (1. - smoothL1_sign)
        out_loss_box = bbox_outside_weights * in_loss_box
        loss_box = tf.reduce_mean(tf.reduce_sum(out_loss_box, axis=dim))
        return loss_box

    def add_losses(self):
        # RPN类别损失
        rpn_cls_score = tf.reshape(self._predictions['rpn_cls_score_reshape'], [-1, 2])
        rpn_label = tf.reshape(self._anchor_targets['rpn_labels'], [-1])
        rpn_select = tf.where(tf.not_equal(rpn_label, -1))
        rpn_cls_score = tf.reshape(tf.gather(rpn_cls_score, rpn_select), [-1, 2])
        rpn_label = tf.reshape(tf.gather(rpn_label, rpn_select), [-1])
        rpn_cross_entropy = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(logits=rpn_cls_score, labels=rpn_label))

        # RPN, bbox loss
        rpn_bbox_pred = self._predictions['rpn_bbox_pred']
        rpn_bbox_targets = self._anchor_targets['rpn_bbox_targets']
        rpn_bbox_inside_weights = self._anchor_targets['rpn_bbox_inside_weights']
        rpn_bbox_outside_weights = self._anchor_targets['rpn_bbox_outside_weights']
        # 待注释：L1平滑（？）
        rpn_loss_box = self._smooth_l1_loss(rpn_bbox_pred, rpn_bbox_targets, rpn_bbox_inside_weights,
                                            rpn_bbox_outside_weights, sigma=3.0, dim=[1, 2, 3])

        # RCNN, class loss
        # RCNN类别损失（待细化）
        cls_score = self._predictions["cls_score"]
        label = tf.reshape(self._proposal_targets["labels"], [-1])
        cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=cls_score, labels=label))

        # RCNN, bbox loss
        # RCNN包围框损失（待细化）
        bbox_pred = self._predictions['bbox_pred']
        bbox_targets = self._proposal_targets['bbox_targets']
        bbox_inside_weights = self._proposal_targets['bbox_inside_weights']
        bbox_outside_weights = self._proposal_targets['bbox_outside_weights']
        # 待注释：L1平滑（？）
        loss_box = self._smooth_l1_loss(bbox_pred, bbox_targets, bbox_inside_weights, bbox_outside_weights)

        self._losses['cross_entropy'] = cross_entropy
        self._losses['loss_box'] = loss_box
        self._losses['rpn_cross_entropy'] = rpn_cross_entropy
        self._losses['rpn_loss_box'] = rpn_loss_box

        loss = cross_entropy + loss_box + rpn_cross_entropy + rpn_loss_box

        # TODO:正则化

        # regularization_loss = tf.losses.get_regularization_loss()
        #
        # # regularization_loss = tf.add_n(tf.losses.get_regularization_losses())
        # self._losses['total_loss'] = loss + regularization_loss

        self._losses['total_loss'] = loss

    def build(self, is_training):
        self.get_anchors()
        self.region_proposal(is_training)
        self.roi_pooling()
        self.head_to_tail(is_training)
        self.region_classification(is_training)
        if is_training:
            self.add_losses()
