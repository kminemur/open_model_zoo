import cv2
import numpy as np

def postprocess(prediction, num_classes, conf_thre = 0.7, nms_thre = 0.45, class_agnostic = False):
    # original source: https://github.com/Megvii-BaseDetection/YOLOX
    # Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
    box_corner = np.zeros(prediction.shape)
    
    # box_corner = prediction.new(prediction.shape)
    box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
    box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
    box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
    box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
    prediction[:, :, :4] = box_corner[:, :, :4]
    output = [None for _ in range(len(prediction))]
    
    for i, image_pred in enumerate(prediction):
        # If none are remaining => process next image
        if len(image_pred[0]) == 0:
            continue
        # Get score and class with highest confidence
        class_conf = np.max(image_pred[:, 5: 5 + num_classes], axis = 1, keepdims = True)
        class_pred = np.argmax(image_pred[:, 5: 5 + num_classes], axis = 1)
        class_pred = np.expand_dims(class_pred, axis = 1)

        conf_mask = (image_pred[:, 4] * class_conf.squeeze() >= conf_thre).squeeze()
        # Detections ordered as (x1, y1, x2, y2, obj_conf, class_conf, class_pred)
        detections = np.concatenate((image_pred[:, :5], class_conf, class_pred), axis=1)
        detections = detections[conf_mask]

        if len(detections)<1:
            continue
        if class_agnostic:
            nms_out_index = nms(
                detections[:, :4],
                detections[:, 4] * detections[:, 5],
                nms_thre)
        else:
            nms_out_index = multiclass_nms(
                detections[:, :4],
                detections[:, 4] * detections[:, 5],
                detections[:, 6],
                nms_thre)

        detections = detections[nms_out_index] # filtering boxes

        if output[i] is None:
            output[i] = detections
        else:
            output[i] = np.concatenate((output[i], detections))
    return output

def nms(boxes, scores, nms_thr):
    """Single class NMS implemented in Numpy."""
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= nms_thr)[0]
        order = order[inds + 1]

    return keep

def multiclass_nms(boxes, scores, nms_thr, score_thr):
    """Multiclass NMS implemented in Numpy"""
    final_dets = []
    num_classes = scores.shape[1]
    for cls_ind in range(num_classes):
        cls_scores = scores[:, cls_ind]
        valid_score_mask = cls_scores > score_thr
        if valid_score_mask.sum() == 0:
            continue
        else:
            valid_scores = cls_scores[valid_score_mask]
            valid_boxes = boxes[valid_score_mask]
            keep = nms(valid_boxes, valid_scores, nms_thr)
            if len(keep) > 0:
                cls_inds = np.ones((len(keep), 1)) * cls_ind
                dets = np.concatenate([valid_boxes[keep], valid_scores[keep, None], cls_inds], 1)
                final_dets.append(dets)
    if len(final_dets) == 0:
        return None
    return np.concatenate(final_dets, 0)

def crop_pad_resize(img: np.ndarray, roi_xyxy: np.ndarray, dsize):
    assert img.ndim == 3 and roi_xyxy.ndim == 1
    assert roi_xyxy.shape[0] == 4

    ### crop image
    rx0, ry0, rx1, ry1 = roi_xyxy.astype(int)
    img_roi = img[ry0: ry1, rx0: rx1]

    ### pad image
    hs, ws = img_roi.shape[:2]
    if ws < hs: # taller image
        padh1 = padh2 = 0
        padw1 = (hs - ws) // 2
        padw2 = hs - (ws + padw1)
        img_res = np.pad(img_roi, ([padh1, padh2], [padw1, padw2], [0, 0]))
    elif ws > hs: # wider image
        padw1 = padw2 = 0
        padh1 = (ws - hs) // 2
        padh2 = ws - (hs + padh1)
        img_res = np.pad(img_roi, ([padh1, padh2], [padw1, padw2], [0, 0]))
    else: # 4 edges identical
        padh1 = padh2 = 0
        padw1 = padw2 = 0
        img_res = img_roi
    hp, wp = img_res.shape[:2]

    ### resize image
    if dsize != hp or dsize != wp:
        img_res = cv2.resize(img_res, (dsize, dsize))

    ### organize & output result
    pad_img_size = (hp, wp)
    pad_left = padw1
    pad_top = padh1

    return img_res, pad_img_size, pad_left, pad_top

def reverse_crop_pad_resize(
    xyxys: np.ndarray,
    pad_img_size: tuple,
    dsize: int,
    pad_left: int,
    pad_top: int,
    roi_xyxy: np.ndarray):
    '''
        reversal of crop_pad_resize
        xyxys: bboxes in xyxy format
        pad_img_size, pad_left, pad_top are from outputs of crop_pad_resize()
        dsize is the resize params used in crop_pad_resize()
        roi_xyxy is the ROI used to crop original image
    '''
    ## resize & un-pad bboxes back to padded image
    hp, wp = pad_img_size
    scalex, scaley = dsize / wp, dsize / hp
    xyxys[:, 0: : 2] = np.clip(xyxys[:, 0: : 2] / scalex - pad_left, 0, wp)
    xyxys[:, 1: : 2] = np.clip(xyxys[:, 1: : 2] / scaley - pad_top, 0, hp)
    ##  un-crop
    offsetx, offsety = roi_xyxy[: 2]
    xyxys[:, 0: : 2] += offsetx
    xyxys[:, 1: : 2] += offsety

    return xyxys
