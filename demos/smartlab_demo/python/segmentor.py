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

import cv2
import numpy as np
import logging as log
from collections import deque
from scipy.special import softmax
from openvino.inference_engine import IECore, StatusCode
import time
from PIL import Image


class Segmentor:
    def __init__(self, ie, device, backbone_path, classifier_path):
        self.terms = [
            "noise_action",
            "put_take",
            "adjust_rider",
        ]

        # Side encoder
        net = ie.read_network(backbone_path)
        self.encSide = ie.load_network(network=net, device_name=device)
        self.encSide_input_keys = list(self.encSide.input_info.keys())
        self.encSide_output_key = list(self.encSide.outputs.keys())
        # Top encoder
        net = ie.read_network(backbone_path)
        self.encTop = ie.load_network(network=net, device_name=device)
        self.encTop_input_keys = list(self.encTop.input_info.keys())
        self.encTop_output_key = list(self.encTop.outputs.keys())
        # Decoder
        net = ie.read_network(classifier_path)
        self.classifier = ie.load_network(network=net, device_name=device)
        self.classifier_input_keys = list(self.classifier.input_info.keys())
        self.classifier_output_key = list(self.classifier.outputs.keys())

        self.shifted_tesor_side = np.zeros(85066)
        self.shifted_tesor_top = np.zeros(85066)

    def inference(self, buffer_top, buffer_side, frame_index):
        """
        Args:
            buffer_top: buffers of the input image arrays for the top view
            buffer_side: buffers of the input image arrays for the side view
            frame_index: frame index of the latest frame
        Returns: the temporal prediction results for each frame (including the historical predictions)，
                 length of predictions == frame_index()
        """

        ### preprocess ###
        buffer_side = buffer_side[120:, :, :]  # remove date characters
        buffer_top = buffer_top[120:, :, :]  # remove date characters
        buffer_side = cv2.resize(buffer_side, (224, 224), interpolation=cv2.INTER_LINEAR)
        buffer_top = cv2.resize(buffer_top, (224, 224), interpolation=cv2.INTER_LINEAR)
        buffer_side = buffer_side / 255
        buffer_top = buffer_top / 255

        buffer_side = buffer_side[np.newaxis, :, :, :].transpose((0, 3, 1, 2)).astype(np.float32)
        buffer_top = buffer_top[np.newaxis, :, :, :].transpose((0, 3, 1, 2)).astype(np.float32)

        ### run ###
        out = self.encSide.infer(
            inputs={self.encSide_input_keys[0]: buffer_side,
                    self.encSide_input_keys[1]: self.shifted_tesor_side})
        feature_vector_side = out[self.encSide_output_key[0]]
        self.shifted_tesor_side = out[self.encSide_output_key[1]]

        out = self.encTop.infer(
            inputs={self.encTop_input_keys[0]: buffer_top,
                    self.encTop_input_keys[1]: self.shifted_tesor_top})
        feature_vector_top = out[self.encTop_output_key[0]]
        self.shifted_tesor_top = out[self.encTop_output_key[1]]

        output = self.classifier.infer(inputs={
            self.classifier_input_keys[0]: feature_vector_side,
            self.classifier_input_keys[1]: feature_vector_top}
        )[self.classifier_output_key[0]]

        ### yoclo classifier ###
        isAction = (output.squeeze()[0] >= .5).astype(int)
        predicted = isAction * (np.argmax(output.squeeze()[1:]) + 1)

        return self.terms[predicted], self.terms[predicted]

    def inference_async(self, buffer_top, buffer_side, frame_index):
        ### preprocess ###
        buffer_side = buffer_side[120:, :, :]  # remove date characters
        buffer_top = buffer_top[120:, :, :]  # remove date characters
        buffer_side = cv2.resize(buffer_side, (224, 224), interpolation=cv2.INTER_LINEAR)
        buffer_top = cv2.resize(buffer_top, (224, 224), interpolation=cv2.INTER_LINEAR)
        buffer_side = buffer_side / 255
        buffer_top = buffer_top / 255

        buffer_side = buffer_side[np.newaxis, :, :, :].transpose((0, 3, 1, 2)).astype(np.float32)
        buffer_top = buffer_top[np.newaxis, :, :, :].transpose((0, 3, 1, 2)).astype(np.float32)

        ### async ###
        self.encSide.requests[0].async_infer(inputs={self.encSide_input_keys[0]: buffer_side,
                                                     self.encSide_input_keys[1]: self.shifted_tesor_side})

        self.encTop.requests[0].async_infer(inputs={self.encTop_input_keys[0]: buffer_top, \
                                                    self.encTop_input_keys[1]: self.shifted_tesor_top})

        while True:
            if not self.encSide.requests[0].wait() and not self.encTop.requests[0].wait():
                feature_vector_side = self.encSide.requests[0].output_blobs[self.encSide_output_key[0]].buffer
                self.shifted_tesor_side = self.encSide.requests[0].output_blobs[self.encSide_output_key[1]].buffer

                feature_vector_top = self.encTop.requests[0].output_blobs[self.encTop_output_key[0]].buffer
                self.shifted_tesor_top = self.encTop.requests[0].output_blobs[self.encTop_output_key[1]].buffer

                output = self.classifier.infer(inputs={
                    self.classifier_input_keys[0]: feature_vector_side,
                    self.classifier_input_keys[1]: feature_vector_top}
                )[self.classifier_output_key[0]]

                ### yoclo classifier ###
                isAction = (output.squeeze()[0] >= .5).astype(int)
                predicted = isAction * (np.argmax(output.squeeze()[1:]) + 1)

                return self.terms[predicted], self.terms[predicted]

    def inference_async_api(self, buffer_top, buffer_side, frame_index):
        predicrted_top, predicrted_side = \
            self.inference_async(buffer_top, buffer_side, frame_index)

        return predicrted_top, predicrted_side


class SegmentorMstcn:
    def __init__(self, ie, device, efficientNet_path, mstcn_path):
        self.ActionTerms = [
            "background",
            "noise_action",
            "remove_support_sleeve",
            "adjust_rider",
            "adjust_nut",
            "adjust_balancing",
            "open_box",
            "close_box",
            "choose_weight",
            "put_left",
            "put_right",
            "take_left",
            "take_right",
            "install_support_sleeve",
        ]

        self.EmbedBufferTop = np.zeros((1280, 0))
        self.EmbedBufferSide = np.zeros((1280, 0))
        self.ImgSizeHeight = 224
        self.ImgSizeWidth = 224
        self.EmbedBatchSize = 1
        self.SegBatchSize = 24
        self.EmbedWindowLength = 1
        self.EmbedWindowStride = 1
        self.EmbedWindowAtrous = 3
        self.TemporalLogits = np.zeros((0, len(self.ActionTerms)))
        self.his_fea = []

        net = ie.read_network(efficientNet_path)
        net.add_outputs("Flatten_237/Reshape")

        self.efficientNet = ie.load_network(network=net, device_name="CPU", num_requests=2)
        self.efficientNet_input_keys = list(self.efficientNet.input_info.keys())
        self.efficientNet_output_key = list(self.efficientNet.outputs.keys())

        self.mstcn_net = ie.read_network(mstcn_path)
        self.mstcn = ie.load_network(network=self.mstcn_net, device_name="CPU")
        self.mstcn_input_keys = list(self.mstcn.input_info.keys())
        self.mstcn_output_key = list(self.mstcn.outputs.keys())
        self.mstcn_net.reshape({'input': (1, 2560, 1)})
        self.reshape_mstcn = ie.load_network(network=self.mstcn_net, device_name=device)
        init_his_feature = np.load('init_his.npz')
        self.his_fea = [init_his_feature['arr_0'],
                        init_his_feature['arr_1'],
                        init_his_feature['arr_2'],
                        init_his_feature['arr_3']]

    def inference(self, buffer_top, buffer_side, frame_index):
        """
        Args:
            buffer_top: buffers of the input image arrays for the top view
            buffer_front: buffers of the input image arrays for the front view
            frame_index: frame index of the latest frame
        Returns: the temporal prediction results for each frame (including the historical predictions),
                 length of predictions == frame_index()
        """
        ### run encoder ###
        self.feature_embedding(
            img_buffer=buffer_top,
            embedding_buffer=self.EmbedBufferTop,
            isTop=0)
        self.feature_embedding(
            img_buffer=buffer_side,
            embedding_buffer=self.EmbedBufferSide,
            isTop=1)

        while True:
            if self.efficientNet.requests[0].wait(0) == StatusCode.OK and self.efficientNet.requests[1].wait(
                0) == StatusCode.OK:
                out_logits_0 = self.efficientNet.requests[0].output_blobs[
                    self.efficientNet_output_key[0]].buffer.transpose(1, 0)
                out_logits_1 = self.efficientNet.requests[1].output_blobs[
                    self.efficientNet_output_key[0]].buffer.transpose(1, 0)
                self.EmbedBufferTop = out_logits_0
                self.EmbedBufferSide = out_logits_1
                self.embedding_buffer = np.concatenate([out_logits_0, out_logits_1],
                                                       axis=1)  # ndarray: C x num_embedding

                ### run mstcn++ ###
                self.action_segmentation()

                # ### get label ###
                valid_index = self.TemporalLogits.shape[0]
                if valid_index == 0:
                    return []
                else:
                    frame_predictions = [self.ActionTerms[i] for i in np.argmax(self.TemporalLogits, axis=1)]
                    frame_predictions = ["background" for i in range(self.EmbedWindowLength - 1)] + frame_predictions

                return frame_predictions[-1]

    def feature_embedding(self, img_buffer, embedding_buffer, isTop=0):
        # minimal temporal length for processor

        num_embedding = embedding_buffer.shape[-1]
        img_buffer = list(img_buffer)
        # absolute index in temporal shaft
        start_index = self.EmbedWindowStride * num_embedding
        input_data = np.array(Image.fromarray(img_buffer[start_index]).resize((224, 224), Image.BILINEAR))
        input_data = np.asarray(input_data).squeeze().transpose(2, 0, 1)
        self.efficientNet.requests[isTop].async_infer({self.efficientNet_input_keys[0]: input_data})

    def action_segmentation(self):
        # read buffer
        embed_buffer_top = self.EmbedBufferTop
        embed_buffer_side = self.EmbedBufferSide
        batch_size = self.SegBatchSize
        start_index = self.TemporalLogits.shape[0]
        end_index = start_index + 1
        num_batch = (end_index - start_index) // batch_size
        if num_batch < 0:
            log.debug("Waiting for the next frame ...")
        elif num_batch == 0:
            log.debug(f"start_index: {start_index} end_index: {end_index}")

            unit1 = embed_buffer_top
            unit2 = embed_buffer_side
            feature_unit = np.concatenate([unit1[:, ], unit2[:, ]], axis=0)
            input_mstcn = np.expand_dims(feature_unit, 0)

            feed_dict = {}
            if len(self.his_fea) != 0:
                feed_dict = {self.mstcn_input_keys[i]: self.his_fea[i] for i in range(4)}
            feed_dict[self.mstcn_input_keys[-1]] = input_mstcn

            if input_mstcn.shape == (1, 2560, 1):
                out = self.reshape_mstcn.infer(inputs=feed_dict)
            else:
                print('shape:', input_mstcn.shape)
            predictions = out[self.mstcn_output_key[-1]]
            self.his_fea = [out[self.mstcn_output_key[i]] for i in range(4)]

            temporal_logits = predictions[:, :, : len(self.ActionTerms), :]  # 4x1x16xN
            temporal_logits = softmax(temporal_logits[-1], 1)  # 1x16xN
            temporal_logits = temporal_logits.transpose((0, 2, 1)).squeeze(axis=0)
            self.TemporalLogits = np.concatenate([self.TemporalLogits, temporal_logits], axis=0)


if __name__ == '__main__':
    ie = IECore()
    segmentor = SegmentorMstcn(ie, "CPU", "E:\\models\\efficientnetb0\\efficientnet-b0-pytorch.xml",
                               "E:\\models\\mstcn_2560\\inferred_model.xml")
    frame_counter = 0  # Frame index counter
    buffer1 = deque(maxlen=1000)  # Array buffer
    buffer2 = deque(maxlen=1000)

    cap1 = cv2.VideoCapture("E:\\video\\P03_A5130001992103255012_2021-10-18_10-19-30_1.mp4")
    cap2 = cv2.VideoCapture("E:\\video\\P03_A5130001992103255012_2021-10-18_10-19-30_2.mp4")
    while cap1.isOpened() and cap2.isOpened():
        ret1, frame1 = cap1.read()  # frame:480 x 640 x 3
        ret2, frame2 = cap2.read()  # frame:480 x 640 x 3
        if ret1 and ret2:
            buffer1.append(cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB))
            buffer2.append(cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB))
            # buffer1.append(frame1)
            # buffer2.append(frame2)
            frame_counter += 1
            start = time.time()
            frame_predictions = segmentor.inference(
                buffer_top=buffer1,
                buffer_side=buffer2,
                frame_index=frame_counter)
            end = time.time()
            print(1 / (end - start))
            print(frame_counter)
            print("Frame predictions:", frame_predictions)
        else:
            print("Finished!")
            break
