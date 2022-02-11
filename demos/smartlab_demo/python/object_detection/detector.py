"""
 Copyright (C) 2021-2022 Intel Corporation

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

import numpy as np
from tkinter import E

from .preprocess import preprocess
from .settings import MwGlobalExp
from .postprocess import postprocess
from .deploy_util import multiclass_nms
from .subdetectors import SubDetector, CascadedSubDetector


class Detector:
    def __init__(self,
        ie,
        device,
        top_models:list,
        side_models:list,
        backend:str='openvino'):

        '''configure output/preview'''
        self.backend = backend

        '''configure settings for 2 models in top view'''
        ###          topview.global_subdetector1           ###
        #  model: mw-xxtop-glb1glb_cls0_yolox-n
        #  max-number constraints:
        #     "balance", 1; "weights",    omit; "tweezers", 1;
        #     "box"    , 1; "battery", 1; "tray"    , 2;
        #     "ruler"  , 1; "rider"  ,    omit; "scale"   , 1;
        #     "hand", 2;  
        #  other conditions:
        #     conf 0.1; nms 0.3
        self.top_glb_exp = MwGlobalExp(
            ie=ie,
            device=device,
            num_classes = 10,
            model_path = top_models[0],
            conf_thresh= 0.1,
            nms_thresh = 0.3)

        ###           topview.scale_subdetector2          ###
        #  model: mw-xxtop-scale1cls4_yolox-n
        #  max-number constraints:
        #     "scale", 1; "roundscrew2", 2;
        #     "pointer", 1; "pointerhead", 1;
        #  other conditions:
        #     conf 0.1; nms 0.2
        self.top_loc_exp = MwGlobalExp(
            ie=ie,
            device=device,
            num_classes = 4,
            model_path = top_models[1],
            conf_thresh= 0.1,
            nms_thresh = 0.2,
            parent_obj = 'scale')
        
        '''configure settings for 2 models in side view'''
        ###          sideview.global_subdetector1           ###
        #  model: mw-xxside-glb1glb_cls0_yolox-n
        #  max-number constraints:
        #     "balance", 1; "weights",    omit; "tweezers", 1;
        #     "box"    , 1; "battery", 1; "tray"    , 2;
        #     "ruler"  , 1; "rider"  ,    omit; "scale"   , 1;
        #     "hand", 2;  
        #  other conditions:
        #     conf 0.2; nms 0.3
        self.side_glb_exp = MwGlobalExp(
            ie=ie,
            device=device,
            num_classes = 10,
            model_path = side_models[0],
            conf_thresh= 0.2,
            nms_thresh = 0.3)

        ###           sideview.ruler_subdetector2          ###
        #  model: mw-xxside-ruler1cls3_yolox-n
        #  max-number constraints:
        #     "ruler", 1; "rider", 1; "roundscrew1", 2;      
        #  other conditions:
        #     conf 0.1; nms 0.3
        self.side_loc_exp = MwGlobalExp(
            ie=ie,
            device=device,
            num_classes = 3,
            model_path = side_models[1],
            conf_thresh= 0.1,
            nms_thresh = 0.3,
            parent_obj = 'ruler')

        ### concatenate list of class names for top/side views
        self.all_classes  = list(self.top_glb_exp.mw_classes)
        self.all_classes += list(self.top_loc_exp.mw_classes)
        self.all_classes += list(self.side_glb_exp.mw_classes)
        self.all_classes += list(self.side_loc_exp.mw_classes)
        self.all_classes = sorted(list(set(self.all_classes)))

        #  max-number constraints:
        self.max_nums = {
        "balance": 1, "weights": 6, "tweezers": 1,
        "box"    : 1, "battery": 1, "tray"    : 2,
        "ruler"  : 1, "rider"  : 1, "scale"   : 1,
        "hand"   : 2, "roundscrew1" : 2, "roundscrew2" : 2,
        "pointer": 1, "pointerhead" : 1}

        ### load models for top view
        self.top_glb_subdetector = SubDetector(self.top_glb_exp, self.all_classes)
        self.top_scale_subdetector = CascadedSubDetector(self.top_loc_exp, self.all_classes)
        ### load models for side view
        self.side_glb_subdetector = SubDetector(self.side_glb_exp, self.all_classes)
        self.side_ruler_subdetector = CascadedSubDetector(self.side_loc_exp, self.all_classes)

    def _get_parent_roi(self, preds, parent_id):
        for pred in preds:
            if parent_id == pred[-1]:
                res = pred[: 4]
                return res
        return None

    def _detect_one(self, img, view='top'):
        if view == 'top': # top view
            glb_subdet = self.top_glb_subdetector
            loc_subdet = self.top_scale_subdetector
        else: # side view
            glb_subdet = self.side_glb_subdetector
            loc_subdet = self.side_ruler_subdetector

        all_preds = []
        for i, sub_detector in enumerate([glb_subdet, loc_subdet]):
            if not hasattr(sub_detector, 'is_cascaded'):
                outputs = sub_detector.inference(img)
            else:
                parent_cat = sub_detector.parent_cat
                parent_id = glb_subdet.detcls2id[parent_cat]
                parent_roi = self._get_parent_roi(all_preds[-1], parent_id)

                if parent_roi is not None:
                    outputs = sub_detector.inference_in(img, parent_roi)
                else:
                    outputs[0] = None
            if outputs[0] is not None:
                preds = outputs[0] # work if bsize = 1
            else:
                continue
            all_preds.append(preds)

        all_preds = np.concatenate(all_preds)
        for r, pred in enumerate(all_preds):
            cls_id = int(pred[-1])
            all_preds[r, -1] = cls_id

        # remap to original image scale
        bboxes = all_preds[:, :4]
        cls = all_preds[:, 6]
        scores = all_preds[:, 4] * all_preds[:, 5]

        return bboxes, cls, scores

    # def _detect_one_async(self, img, view='top'):
    #     if view == 'top': # top view
    #         sub_detector1 = self.top1_subdetector
    #         sub_detector2 = self.top2_subdetector
    #     else: # front view
    #         sub_detector1 = self.front1_subdetector
    #         sub_detector2 = self.front2_subdetector

    #     ### openvino async_mode ###
    #     exec_net1, img_info1 = sub_detector1.inference_async(img)
    #     exec_net2, img_info2 = sub_detector2.inference_async(img)

    #     return exec_net1, exec_net2, img_info1

    # def _detect_one_wait(self, exec_net1, exec_net2, img_info, view='top'):
    #     if view == 'top': # top view
    #         sub_detector1 = self.top1_subdetector
    #         sub_detector2 = self.top2_subdetector
    #     else: # front view
    #         sub_detector1 = self.front1_subdetector
    #         sub_detector2 = self.front2_subdetector

    #     all_preds = []
    #     while True:
    #         if not exec_net1.requests[0].wait() and not exec_net2.requests[0].wait():
    #             res1 = exec_net1.requests[0].output_blobs[sub_detector1.onode].buffer
    #             res2 = exec_net2.requests[0].output_blobs[sub_detector2.onode].buffer

    #             import time
    #             outputs1 = demo_postprocess(res1, sub_detector1.input_shape, p6=False)
    #             outputs2 = demo_postprocess(res2, sub_detector2.input_shape, p6=False)

    #             outputs1 = postprocess(
    #                 outputs1, sub_detector1.num_classes, sub_detector1.conf_thresh,
    #                 sub_detector1.nms_thresh, class_agnostic=True)
    #             outputs2 = postprocess(
    #                 outputs2, sub_detector2.num_classes, sub_detector2.conf_thresh,
    #                 sub_detector2.nms_thresh, class_agnostic=True)

    #             if outputs1[0] is not None:
    #                 preds1 = outputs1[0]
    #                 preds1[:, 6] += self.offset_cls_idx[0]
    #                 all_preds.append(preds1)
    #             if outputs2[0] is not None:
    #                 preds2 = outputs2[0]
    #                 preds2[:, 6] += self.offset_cls_idx[1]
    #                 all_preds.append(preds2)

    #             if len(all_preds) > 0:
    #                 all_preds = np.concatenate(all_preds)
    #             else:
    #                 all_preds = np.zeros((1, 7))

    #             # merge same classes from model 2
    #             for r, pred in enumerate(all_preds):
    #                 cls_id = int(pred[-1])
    #                 if cls_id in self.repeat_cls2_ids:
    #                     all_preds[r, -1] = self.cls2tocls1[cls_id]

    #             # restrict object number for each class
    #             all_preds = self._apply_detection_constraints(all_preds)

    #             # remap to original image scale
    #             ratio = img_info['ratio'] # all ways same?
    #             bboxes = all_preds[:, :4] / ratio
    #             cls = all_preds[:, 6]
    #             scores = all_preds[:, 4] * all_preds[:, 5]

    #             return bboxes, cls, scores

    def inference(self, img_top, img_side):
        """
        Given input arrays for two view, need to generate and save 
            the corresponding detection results in the specific data structure.
        Args:
        img_top: img array of H x W x C for the top view
        img_front: img_array of H x W x C for the front view

        Returns:
        prediction results for the two images
        """

        ### sync mode ###
        top_bboxes, top_cls_ids, top_scores = self._detect_one(img_top, view='top')
        side_bboxes, side_cls_ids, side_scores = self._detect_one(img_side, view='side')

        # get class label
        top_labels = [self.all_classes[int(i)-1] for i in top_cls_ids]
        side_labels = [self.all_classes[int(i)-1] for i in side_cls_ids]

        return [top_bboxes, top_cls_ids, top_labels, top_scores], [side_bboxes, side_cls_ids, side_labels, side_scores]

    # def inference_async(self, img_top, img_front):
    #     """
    #     todo Given input arrays for two view, need to generate and save the corresponding detection results
    #         in the specific data structure.
    #     Args:
    #     img_top: img array of H x W x C for the top view
    #     img_front: img_array of H x W x C for the front view

    #     Returns:
    #     prediction results for the two images
    #     """

    #     ### Async mode ###
    #     exec_net1, exec_net2, img_info1 = self._detect_one_async(img_top, view='top')
    #     exec_net3, exec_net4, img_info2 = self._detect_one_async(img_front, view='front')

    #     top_bboxes, top_cls_ids, top_scores = self._detect_one_wait(exec_net1, exec_net2, img_info1, view='top')
    #     front_bboxes, front_cls_ids, front_scores = self._detect_one_wait(exec_net3, exec_net4, img_info2, view='front')

    #     # get class string
    #     top_cls_ids = [ self.classes[int(x)] for x in top_cls_ids ]
    #     front_cls_ids = [ self.classes[int(x)] for x in front_cls_ids ]

    #     # return [], []
    #     return [top_bboxes, top_cls_ids, top_scores], [front_bboxes, front_cls_ids, front_scores]

    # def inference_async_api(self, img_top, img_front):
    #     top_det_results, front_det_results = \
    #         self.inference_async(img_top, img_front)

    #     return top_det_results, front_det_results