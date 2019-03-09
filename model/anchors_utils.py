import tensorflow as tf
import numpy as np


def _get_w_h_ctrs(anchor):
    """
    返回窗口的宽，高，中心点x，中心点y
    """
    w = anchor[2] - anchor[0] + 1
    h = anchor[3] - anchor[1] + 1
    x_ctr = anchor[0] + 0.5 * (w - 1)
    y_ctr = anchor[1] + 0.5 * (h - 1)
    return w, h, x_ctr, y_ctr


def _make_anchors(ws, hs, x_ctr, y_ctr):
    ws = ws[:, np.newaxis]
    hs = hs[:, np.newaxis]
    # 生成的anchor为np数组
    # anchor坐标为左上xy，右下xy
    anchors = np.hstack((x_ctr - 0.5 * (ws - 1),
                         y_ctr - 0.5 * (hs - 1),
                         x_ctr + 0.5 * (ws - 1),
                         y_ctr + 0.5 * (hs - 1)))
    return anchors


def _ratio_enum(anchor, ratios):
    w, h, x_ctr, y_ctr = _get_w_h_ctrs(anchor)
    size = w * h
    size_ratios = size / ratios
    ws = np.round(np.sqrt(size_ratios))
    hs = np.round(ws * ratios)
    anchors = _make_anchors(ws, hs, x_ctr, y_ctr)
    return anchors


def _scale_enum(anchor, scales):
    """
    Enumerate a set of anchors for each scale wrt an anchor.
    """
    # 缩放枚举
    w, h, x_ctr, y_ctr = _get_w_h_ctrs(anchor)
    ws = w * scales
    hs = h * scales
    anchors = _make_anchors(ws, hs, x_ctr, y_ctr)
    return anchors


def get_basic_anchors(anchor_ratios, anchor_scales, base_size=16):
    base_anchor = np.array([1, 1, base_size, base_size]) - 1
    ratio_anchors = _ratio_enum(base_anchor, anchor_ratios)
    anchors = np.vstack([_scale_enum(ratio_anchors[i, :], anchor_scales) for i in range(ratio_anchors.shape[0])])
    return anchors


def generate_anchors(height_of_feature_map, width_of_feature_map, stride_of_feature_map, anchor_scales, anchor_ratios):
    shift_x = tf.range(width_of_feature_map) * stride_of_feature_map
    shift_y = tf.range(height_of_feature_map) * stride_of_feature_map
    shift_x, shift_y = tf.meshgrid(shift_x, shift_y)
    sx = tf.reshape(shift_x, shape=(-1,))
    sy = tf.reshape(shift_y, shape=(-1,))
    shifts = tf.transpose(tf.stack([sx, sy, sx, sy]))
    num_of_feature_map_pixels = tf.multiply(width_of_feature_map, height_of_feature_map)
    shifts = tf.transpose(tf.reshape(shifts, shape=[1, num_of_feature_map_pixels, 4]), perm=(1, 0, 2))
    basic_anchors = get_basic_anchors(anchor_ratios=np.array(anchor_ratios), anchor_scales=np.array(anchor_scales))
    num_of_basic_anchors = basic_anchors.shape[0]
    anchor_constant = tf.constant(basic_anchors.reshape((1, num_of_basic_anchors, 4)), dtype=tf.int32)
    total_num_of_anchors = num_of_basic_anchors * num_of_feature_map_pixels
    anchors = tf.reshape(tf.add(anchor_constant, shifts), shape=(total_num_of_anchors, 4))
    return tf.cast(anchors, dtype=tf.float32), total_num_of_anchors
