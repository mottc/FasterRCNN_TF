import numpy as np


def bbox_overlaps(anchors, gt_boxes):
    N = anchors.shape[0]
    K = gt_boxes.shape[0]

    overlaps = np.zeros((N, K), dtype=np.float32)
    for k in range(K):
        box_area = ((gt_boxes[k, 2] - gt_boxes[k, 0] + 1) * (gt_boxes[k, 3] - gt_boxes[k, 1] + 1))
        for n in range(N):
            iw = (min(anchors[n, 2], gt_boxes[k, 2]) - max(anchors[n, 0], gt_boxes[k, 0]) + 1)
            if iw > 0:
                ih = (min(anchors[n, 3], gt_boxes[k, 3]) - max(anchors[n, 1], gt_boxes[k, 1]) + 1)
                if ih > 0:
                    ua = float((anchors[n, 2] - anchors[n, 0] + 1) * (anchors[n, 3] - anchors[n, 1] + 1) + box_area - iw * ih)
                    overlaps[n, k] = iw * ih / ua
    return overlaps
