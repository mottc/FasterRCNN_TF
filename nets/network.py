import os
import tensorflow as tf
import net_config
from nets.anchors_utils import generate_anchors
from nets.target_layer import anchor_target_layer, proposal_target_layer


class Network(object):
    def __init__(self, num_classes):
        self._predictions = {}
        self._anchor_targets = {}
        self._proposal_targets = {}
        self._losses = {}
        self._num_anchors = len(net_config.ANCHOR_SCALES) * len(net_config.ANCHOR_RATIOS)
        self._num_classes = num_classes
        self.build(True)

    def get_data(self):
        def _parse_function(image_name, gt_boxes, image_info):
            image_string = tf.read_file(image_name)
            image_decoded = tf.image.decode_jpeg(image_string)  # TODO:数据格式
            # TODO:提取图片尺寸,not here！！
            image_resized = tf.image.resize_images(image_decoded, net_config.RESIZED_IMAGE_SIZE)
            # FIXME:从gt_boxes_names到gt_boxes，从image_info_names到image_info
            return image_resized, gt_boxes, image_info

        img_names = os.listdir('./train')  # TODO:将路径提取出来
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
        dataset = dataset.shuffle(buffer_size=50000)  # TODO:提取buffer_size大小
        dataset = dataset.repeat()
        dataset = dataset.batch(1)

        iterator = dataset.make_one_shot_iterator()
        image_data, gt_boxes, image_info = iterator.get_next()
        return image_data, gt_boxes, image_info

    def get_feature_map(self, image_data):
        pass

    def get_anchors(self, image_info):
        # TODO:这个16是原图到feature map缩小的倍数,提出来？
        height_of_feature_map = tf.to_int32(tf.ceil(image_info[0] / tf.to_float(16)))
        width_of_feature_map = tf.to_int32(tf.ceil(image_info[1] / tf.to_float(16)))
        anchors, num_of_anchors = generate_anchors(height_of_feature_map, width_of_feature_map, 16,
                                                   net_config.ANCHOR_SCALES,
                                                   net_config.ANCHOR_RATIOS)
        return anchors, num_of_anchors

    def _proposal_target_layer(self, rois, roi_scores, name):
        with tf.variable_scope(name) as scope:
            rois, roi_scores, labels, bbox_targets, bbox_inside_weights, bbox_outside_weights = tf.py_func(
                proposal_target_layer,
                [rois, roi_scores, self._gt_boxes, self._num_classes],
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
                [rpn_class_score, self.gt_boxes, self.image_info, 16, self.anchors, self._num_anchors],
                [tf.float32, tf.float32, tf.float32, tf.float32],
                name="anchor_target")

            rpn_labels.set_shape([1, 1, None, None])
            rpn_bbox_targets.set_shape([1, None, None, self._num_anchors * 4])
            rpn_bbox_inside_weights.set_shape([1, None, None, self._num_anchors * 4])
            rpn_bbox_outside_weights.set_shape([1, None, None, self._num_anchors * 4])

            rpn_labels = tf.to_int32(rpn_labels, name="to_int32")
            self._anchor_targets['rpn_labels'] = rpn_labels
            self._anchor_targets['rpn_bbox_targets'] = rpn_bbox_targets
            self._anchor_targets['rpn_bbox_inside_weights'] = rpn_bbox_inside_weights
            self._anchor_targets['rpn_bbox_outside_weights'] = rpn_bbox_outside_weights

        return rpn_labels

    def region_proposal(self, feature_map, is_training):
        def _reshape_layer(bottom, num_dim, name):
            input_shape = tf.shape(bottom)
            with tf.variable_scope(name) as scope:
                to_caffe = tf.transpose(bottom, [0, 3, 1, 2])
                reshaped = tf.reshape(to_caffe, tf.concat(axis=0, values=[[1, num_dim, -1], [input_shape[2]]]))
                to_tf = tf.transpose(reshaped, [0, 2, 3, 1])
                return to_tf

        def _softmax_layer(self, bottom, name):
            if name.startswith('rpn_cls_prob_reshape'):
                input_shape = tf.shape(bottom)
                bottom_reshaped = tf.reshape(bottom, [-1, input_shape[-1]])
                reshaped_score = tf.nn.softmax(bottom_reshaped, name=name)
                return tf.reshape(reshaped_score, input_shape)
            return tf.nn.softmax(bottom, name=name)

        def bbox_transform_inv(boxes, deltas):

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

        def clip_boxes(boxes, im_info):
            # 按照图像大小裁剪boxes
            b0 = tf.maximum(tf.minimum(boxes[:, 0], im_info[1] - 1), 0)
            b1 = tf.maximum(tf.minimum(boxes[:, 1], im_info[0] - 1), 0)
            b2 = tf.maximum(tf.minimum(boxes[:, 2], im_info[1] - 1), 0)
            b3 = tf.maximum(tf.minimum(boxes[:, 3], im_info[0] - 1), 0)
            return tf.stack([b0, b1, b2, b3], axis=1)

        rpn = tf.layers.conv2d(inputs=feature_map, filters=512, kernel_size=[3, 3], padding='SAME',
                               trainable=is_training)
        rpn_class_score = tf.layers.conv2d(rpn, self._num_anchors * 2, [1, 1], trainable=is_training)  # H*W*18
        rpn_cls_score_reshape = _reshape_layer(rpn_class_score, 2, 'rpn_cls_score_reshape')
        rpn_cls_prob_reshape = _softmax_layer(rpn_cls_score_reshape, "rpn_cls_prob_reshape")
        rpn_cls_pred = tf.argmax(tf.reshape(rpn_cls_score_reshape, [-1, 2]), axis=1, name="rpn_cls_pred")
        rpn_cls_prob = _reshape_layer(rpn_cls_prob_reshape, self._num_anchors * 2, "rpn_cls_prob")
        rpn_bbox_pred = tf.layers.conv2d(rpn, self._num_anchors * 4, [1, 1], trainable=is_training)

        self._predictions["rpn_cls_score_reshape"] = rpn_cls_score_reshape

        if is_training:
            scores = rpn_cls_prob[:, :, :, self._num_anchors:]
            scores = tf.reshape(scores, shape=(-1,))
            rpn_bbox_pred = tf.reshape(rpn_bbox_pred, shape=(-1, 4))

            proposals = bbox_transform_inv(self.anchors, rpn_bbox_pred)
            proposals = clip_boxes(proposals, self.image_info[:2])
            # TODO:test300,train2000。提取
            indices = tf.image.non_max_suppression(proposals, scores, max_output_size=2000 - 300, iou_threshold=0.7)
            boxes = tf.gather(proposals, indices)
            boxes = tf.to_float(boxes)
            scores = tf.gather(scores, indices)
            scores = tf.reshape(scores, shape=(-1, 1))
            # 在每个indices前加入batch内索引，由于目前仅支持每个batch一张图像作为输入所以均为0
            batch_inds = tf.zeros((tf.shape(indices)[0], 1), dtype=tf.float32)
            rois = tf.concat([batch_inds, boxes], 1)
            rois.set_shape([None, 5])
            scores.set_shape([None, 1])

            rpn_labels = self._anchor_target_layer(rpn_class_score, "anchor")
            with tf.control_dependencies([rpn_labels]):
                rois, _ = self._proposal_target_layer(rois, scores, "rpn_rois")

        else:
            scores = rpn_cls_prob[:, :, :, self._num_anchors:]
            scores = tf.reshape(scores, shape=(-1,))
            rpn_bbox_pred = tf.reshape(rpn_bbox_pred, shape=(-1, 4))

            proposals = bbox_transform_inv(self.anchors, rpn_bbox_pred)
            proposals = clip_boxes(proposals, self.image_info[:2])
            # TODO:test300,train2000。提取
            indices = tf.image.non_max_suppression(proposals, scores, max_output_size=300, iou_threshold=0.7)
            boxes = tf.gather(proposals, indices)
            boxes = tf.to_float(boxes)
            scores = tf.gather(scores, indices)
            scores = tf.reshape(scores, shape=(-1, 1))
            # 在每个indices前加入batch内索引，由于目前仅支持每个batch一张图像作为输入所以均为0
            batch_inds = tf.zeros((tf.shape(indices)[0], 1), dtype=tf.float32)
            rois = tf.concat([batch_inds, boxes], 1)
            rois.set_shape([None, 5])
            scores.set_shape([None, 1])

        self._predictions["rpn_cls_score"] = rpn_class_score
        self._predictions["rpn_cls_score_reshape"] = rpn_cls_score_reshape
        self._predictions["rpn_cls_prob"] = rpn_cls_prob
        self._predictions["rpn_cls_pred"] = rpn_cls_pred
        self._predictions["rpn_bbox_pred"] = rpn_bbox_pred
        self._predictions["rois"] = rois
        return rois

    def roi_pooling(self, feature_map, rois):
        batch_ids = tf.squeeze(tf.slice(rois, [0, 0], [-1, 1], name="batch_id"), [1])
        # Get the normalized coordinates of bounding boxes
        # 获取包围框归一化后的坐标系（待细化）
        bottom_shape = tf.shape(feature_map)
        height = (tf.to_float(bottom_shape[1]) - 1.) * tf.to_float(16)  # TODO:
        width = (tf.to_float(bottom_shape[2]) - 1.) * tf.to_float(16)
        x1 = tf.slice(rois, [0, 1], [-1, 1], name="x1") / width
        y1 = tf.slice(rois, [0, 2], [-1, 1], name="y1") / height
        x2 = tf.slice(rois, [0, 3], [-1, 1], name="x2") / width
        y2 = tf.slice(rois, [0, 4], [-1, 1], name="y2") / height

        bboxes = tf.stop_gradient(tf.concat([y1, x1, y2, x2], axis=1))
        # POOLING_SIZE 池化后区域大小
        pre_pool_size = 7 * 2  # TODO:
        # 剪裁并通过插值方法调整尺寸
        crops = tf.image.crop_and_resize(feature_map, bboxes, tf.to_int32(batch_ids), [pre_pool_size, pre_pool_size],
                                         name="crops")

        return tf.layers.max_pooling2d(crops, [2, 2], 2, 'same')

    def head_to_tail(self, pool, is_training):
        pass

    def region_classification(self, fc, is_training):

        cls_score = tf.layers.dense(fc, self._num_classes, trainable=is_training)
        cls_prob = tf.nn.softmax(cls_score)
        cls_pred = tf.argmax(cls_score, axis=1)
        bbox_pred = tf.layers.dense(fc, self._num_classes * 4, trainable=is_training)

        self._predictions["cls_score"] = cls_score
        self._predictions["cls_pred"] = cls_pred
        self._predictions["cls_prob"] = cls_prob
        self._predictions["bbox_pred"] = bbox_pred

        return cls_prob, bbox_pred

    def _smooth_l1_loss(self, bbox_pred, bbox_targets, bbox_inside_weights, bbox_outside_weights, sigma=1.0, dim=[1]):
        sigma_2 = sigma ** 2
        box_diff = bbox_pred - bbox_targets
        in_box_diff = bbox_inside_weights * box_diff
        abs_in_box_diff = tf.abs(in_box_diff)
        smoothL1_sign = tf.stop_gradient(tf.to_float(tf.less(abs_in_box_diff, 1. / sigma_2)))
        in_loss_box = tf.pow(in_box_diff, 2) * (sigma_2 / 2.) * smoothL1_sign + (abs_in_box_diff - (0.5 / sigma_2)) * (
                1. - smoothL1_sign)
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
        regularization_loss = tf.add_n(tf.losses.get_regularization_losses(), 'regu')
        self._losses['total_loss'] = loss + regularization_loss

    def build(self, is_training):
        self.image_data, self.gt_boxes, self.image_info = self.get_data()
        feature_map = self.get_feature_map(self.image_data)
        self.anchors, self.anchor_length = self.get_anchors(self.image_info)
        rois = self.region_proposal(feature_map)
        pool = self.roi_pooling(feature_map, rois)
        fc = self.head_to_tail(pool, is_training)
        cls_prob, bbox_pred = self.region_classification(fc, is_training)
        if is_training:
            self.add_losses()
