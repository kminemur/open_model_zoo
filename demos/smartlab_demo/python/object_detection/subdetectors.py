import numpy as np

from .settings import MwGlobalExp
from .preprocess import preprocess
from .deploy_util import ov_postorganize
from .postprocess import postprocess, crop_pad_resize, reverse_crop_pad_resize

class SubDetector:
    def __init__(self, exp: MwGlobalExp, all_classes: list):
        self.inode, self.onode, self.input_shape, self.model = exp.get_openvino_model()

        ### create bi-directional dictionary on all classes
        self.all_classes = all_classes # class list in detector
        self.detcls2id = {c: i+1 for i, c in enumerate(all_classes)} # detector class mapper: cls -> glb_id
        ### create bi-directional dictionary on sub classes
        self.sub_classes = list(exp.mw_classes) # class list in sub-detector
        self.subdetcls2id = {c: i+1 for i, c in enumerate(self.sub_classes)} # sub-detector class mapper: cls -> loc_id
        ### create subdet_id -> det_id dictionary
        self.subdetid2detid = {} # class idx mapper: subdet_id -> det_id
        for i, c in enumerate(self.sub_classes):
            subdet_id = i + 1
            det_id = self.detcls2id[c]
            self.subdetid2detid[subdet_id] = det_id

        self.n_sub_classes = exp.num_classes
        self.num_classes = exp.num_classes
        self.conf_thresh = exp.conf_thresh
        self.nms_thresh = exp.nms_thresh

    def inference(self, img):
        img_info = {"id": 0}
        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width

        ### preprocessing
        img_feed, ratio = preprocess(img, self.input_shape)
        img_info["ratio"] = ratio
        ### model infer
        res = self.model.infer(inputs={self.inode:img_feed})[self.onode]
        ### post-adjustment and NMS
        outputs = ov_postorganize(res, self.input_shape, p6 = False)
        outputs = postprocess(
            outputs, self.num_classes, self.conf_thresh,
            self.nms_thresh, class_agnostic = True)

        ### map cls idx back to global dictionary
        outputs = outputs[0]
        if outputs is None:
            return [None]
        subdet_ids = [int(v+1) for v in outputs[:, -1]]
        det_ids = [self.subdetid2detid[i] for i in subdet_ids]
        outputs[:, -1] = np.array(det_ids, dtype = outputs.dtype)
        outputs[:, :4] /= ratio

        return [outputs]

    # def inference_async(self, img):
    #     img_info = {"id": 0}
    #     height, width = img.shape[:2]
    #     img_info["height"] = height
    #     img_info["width"] = width

    #     img_feed, ratio = preprocess(img, self.input_shape)
    #     img_info["ratio"] = ratio
    #     # res = self.model.infer(inputs={self.inode:img_feed})[self.onode]
    #     self.model.requests[0].async_infer(inputs={self.inode:img_feed})

    #     return self.model, img_info

    def pseudolabel(self, output, img_info, idx_offset, cls_conf = 0.35):
        ratio = img_info["ratio"]
        img = img_info["raw_img"]
        image_id = img_info['id']
        if output is None:
            return img
        bboxes = output[:, 0: 4]
        # preprocessing: resize
        bboxes /= ratio # [[x0,y0,x1,y1], ...]
        cls = output[:, 6]
        scores = output[:, 4] * output[:, 5]
        # enumerate all bbox
        i = 0
        res = []
        for box, c, s in zip(bboxes, cls, scores):
            if s < cls_conf:
                continue
            x0, y0, x1, y1 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            w, h = x1 - x0, y1 - y0
            idx = idx_offset + i
            i += 1
            cat_id = int(c)
            res.append({
                'area': w * h, 
                'bbox': [x0, y0, w, h],
                'category_id': cat_id,
                'id': idx,
                'image_id': image_id,
                'iscrowd': 0,
                'segmentation': [[x0, y0, x1, y1]]})

        return res

class CascadedSubDetector(SubDetector):
    def __init__(self, exp: MwGlobalExp, all_classes: list):
        super().__init__(exp, all_classes)

        self.dsize = exp.input_size[0]
        self.is_cascaded = True # used to determine if this is CascadedDetector
        ### save parent object info
        self.parent_cat = exp.parent_cat
        ### save children object infos
        self.children_cats = exp.children_cats

    def inference_in(self, img, roi_xyxy):
        ### cook raw image => sub-img within ROI of parent object 
        roi_xyxy = roi_xyxy.astype(int)
        img_resize, img_pad_size, pad_left, pad_top = \
            crop_pad_resize(img, roi_xyxy, self.dsize)
        outputs = self.inference(img_resize)
        outputs = outputs[0]
        if outputs is None:
            return [None]

        ### map bbox coords back to raw image
        bboxes_xyxy = outputs[:, : 4]
        outputs[:, : 4] = reverse_crop_pad_resize(
            bboxes_xyxy,
            img_pad_size,
            self.dsize,
            pad_left,
            pad_top,
            roi_xyxy)

        return [outputs]
