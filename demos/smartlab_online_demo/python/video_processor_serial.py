# -*- coding: utf-8 -*-

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

import os
import cv2
import pandas as pd
import numpy as np
import time
from collections import deque
import threading
from object_detection.detector import Detector
from temporal_segmentation.segmentor import Segmentor
from score_evaluation.evaluator import Evaluator
from display.display import Display


class Application(object):
    """
        Video process for the pre-recorder video (in this version we aim to mimic the online process )
    """
    def __init__(self, ):
        """
            Initialize Variables
        """
        self.playing = True  # Control button for video processing
        self.frame_counter =  0 # Frame index counter
        self.buffer_top = deque(maxlen=1000)  # Array buffer
        self.buffer_front = deque(maxlen=1000)

        ''' Progress Variables'''
        self.det_process_counter = 0  # Number of frames processed by the object detection module
        self.seg_process_counter = 0  # Number of frames processed by the temporal segmentation module
        self.eval_process_counter = 0  # Number of frames processed by the score evaluation module

        ''' Object Detection Variables'''
        self.detector = Detector(["./intel/smartlab-object-detection-0001/FP32-INT1/mw-topview-all-yolox-n.bin",
                "./intel/smartlab-object-detection-0002/FP32-INT1/mw-topview-move-yolox-n.bin"],
                ["./intel/smartlab-object-detection-0003/FP32-INT1/mw-frontview-all-yolox-n.bin",
                "./intel/smartlab-object-detection-0004/FP32-INT1/mw-frontview-move-yolox-n.bin"],
                False)
        self.detector.initialize()  # Initialize the session and load the model parameters

        '''Video Segmentation Variables'''
        self.segmentor = Segmentor(
                "./intel/smartlab-action-recognition-encoder-0001/FP32-INT1/1280vec-mobilenet-v2.pt",
                "./intel/smartlab-action-recognition-decoder-0001/FP32-INT1/concat-classifier.pth")
        self.segmentor.initialize()  # Initialize the session and load the model parameters


        '''Score Evaluation Variables'''
        self.evaluator = Evaluator()
        self.evaluator.initialize()

        '''Display Obj Detection, Action Segmentation and Score Evaluation Result'''
        self.display = Display()
        self.display.initialize()

    def video_parser(self, top_video_path, front_video_path):
        """
            Process the video.
        """
        self.cap_top = cv2.VideoCapture(top_video_path)
        self.cap_front = cv2.VideoCapture(front_video_path)

        while self.cap_top.isOpened() and self.cap_front.isOpened() and self.playing:
            ret_top, frame_top = self.cap_top.read()  # frame:480 x 640 x 3
            ret_front, frame_front = self.cap_front.read()

            if ret_top and ret_front:
                self.buffer_top.append(cv2.cvtColor(frame_top, cv2.COLOR_BGR2RGB))
                self.buffer_front.append(cv2.cvtColor(frame_front, cv2.COLOR_BGR2RGB))
                self.frame_counter += 1

                ''' The object detection module need to generate detection results(for the current frame) '''
                top_det_results, front_det_results = self.detector.inference(img_top=frame_top, img_front=frame_front,frame_counter=self.frame_counter)

                ''' The temporal segmentation module need to self judge and generate segmentation results for all historical frames '''
                top_seg_results, front_seg_results = self.segmentor.inference(buffer_top=self.buffer_top,
                                                                            buffer_front=self.buffer_front,
                                                                            frame_index=self.frame_counter
                                                                            )

                ''' The score evaluation module need to merge the results of the two modules and generate the scores '''
                self.state, self.scoring = self.evaluator.inference(top_det_results=top_det_results,
                                                                front_det_results=front_det_results,
                                                                top_seg_results=top_seg_results,
                                                                front_seg_results=front_seg_results,
                                                                frame_top=frame_top,
                                                                frame_front=frame_front
                                                                )

                self.display.display_result(frame_top=frame_top,
                                            frame_front=frame_front,
                                            front_seg_results=front_seg_results,
                                            top_seg_results=top_seg_results,
                                            top_det_results=top_det_results,
                                            front_det_results=front_det_results,
                                            scoring=self.scoring,
                                            state=self.state,
                                            frame_counter=self.frame_counter)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):     #press 'q' to exit
                    break
            else:
                print("Finished !")
                break

    def get_video_fps(self, cap):
        return cap.get(cv2.CAP_PROP_FPS)

    def get_video_total_frames(self, cap):
        return cap.get(cv2.CAP_PROP_FRAME_COUNT)

if __name__ == "__main__":
    application = Application()
    application.video_parser(top_video_path="./2.mp4",
                             front_video_path="./3.mp4")