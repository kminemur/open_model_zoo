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
from scipy.special import softmax


class Segmentor:
    def __init__(self, ie, device, backbone_path, classifier_path):
        self.terms = [
            "noise_action",
            "put_take",
            "adjust_rider",
        ]

        # Side encoder
        net = ie.read_network(backbone_path)
        self.encSide = ie.load_network(network = net, device_name=device)
        self.encSide_input_keys = list(self.encSide.input_info.keys())
        self.encSide_output_key = list(self.encSide.outputs.keys())
        # Top encoder
        net = ie.read_network(backbone_path)
        self.encTop = ie.load_network(network=net, device_name = device)
        self.encTop_input_keys = list(self.encTop.input_info.keys())
        self.encTop_output_key = list(self.encTop.outputs.keys())
        # Decoder
        net = ie.read_network(classifier_path)
        self.classifier = ie.load_network(network = net, device_name = device)
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
        buffer_side = buffer_side[120:, :, :] # remove date characters
        buffer_top = buffer_top[120:, :, :] # remove date characters
        buffer_side = cv2.resize(buffer_side, (224, 224), interpolation = cv2.INTER_LINEAR)
        buffer_top = cv2.resize(buffer_top, (224, 224), interpolation = cv2.INTER_LINEAR)
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
        buffer_side = buffer_side[120:, :, :] # remove date characters
        buffer_top = buffer_top[120:, :, :] # remove date characters
        buffer_side = cv2.resize(buffer_side, (224, 224), interpolation = cv2.INTER_LINEAR)
        buffer_top = cv2.resize(buffer_top, (224, 224), interpolation = cv2.INTER_LINEAR)
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
    def __init__(self, ie, device, encoder_path, mstcn_path):
        self.ActionTerms = [
            "background",
            "noise_action",
            "remove_support_sleeve",
            "remove_pointer_sleeve",
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
            "install_pointer_sleeve",
        ]

        self.EmbedBufferTop = np.zeros((1280, 0))
        self.EmbedBufferFront = np.zeros((1280, 0))
        self.ImgSizeHeight = 224
        self.ImgSizeWidth = 224
        self.EmbedBatchSize = 1
        self.SegBatchSize = 24
        self.EmbedWindowLength = 1
        self.EmbedWindowStride = 1
        self.EmbedWindowAtrous = 3
        self.TemporalLogits = np.zeros((0, len(self.ActionTerms)))

        net = ie.read_network(encoder_path)
        net.add_outputs("Flatten_237/Reshape")

        self.encoder = ie.load_network(network = net, device_name = device)
        self.encoder_input_keys = list(self.encoder.input_info.keys())
        self.encoder_output_key = list(self.encoder.outputs.keys())

        self.mstcn_net = ie.read_network(mstcn_path)
        self.mstcn = ie.load_network(network = self.mstcn_net, device_name = device)
        self.mstcn_input_keys = list(self.mstcn.input_info.keys())
        self.mstcn_output_key = list(self.mstcn.outputs.keys())
        self.mstcn_net.reshape({'input': (1, 2560, 1)})
        self.reshape_mstcn = ie.load_network(network = self.mstcn_net, device_name = device)
        init_his_feature = np.load('init_his.npz')
        self.his_fea = [init_his_feature['arr_0'],
                init_his_feature['arr_1'],
                init_his_feature['arr_2'],
                init_his_feature['arr_3']]

    def inference(self, buffer_top, buffer_side, frame_index):
        """
        Args:
            buffer_top: buffers of the input image arrays for the top view
            buffer_side: buffers of the input image arrays for the side view
            frame_index: frame index of the latest frame
        Returns: the temporal prediction results for each frame (including the historical predictions)，
                 length of predictions == frame_index()
        """
        ### run encoder ###
        self.EmbedBufferTop = self.feature_embedding(
            img_buffer=buffer_top,
            embedding_buffer=self.EmbedBufferTop,
            frame_index=frame_index)
        self.EmbedBufferSide = self.feature_embedding(
            img_buffer=buffer_side,
            embedding_buffer=self.EmbedBufferSide,
            frame_index=frame_index)

        ### run mstcn++ only batch size 1###
        if min(self.EmbedBufferTop.shape[-1], self.EmbedBufferSide.shape[-1]) > 0:
            self.action_segmentation()

        # ### get label ###
        valid_index = self.TemporalLogits.shape[0]
        if valid_index == 0:
            return []
        else:
            frame_predictions = [self.ActionTerms[i] for i in np.argmax(self.TemporalLogits, axis = 1)]
            frame_predictions = ["background" for i in range(self.EmbedWindowLength - 1)] + frame_predictions

        return frame_predictions[-1]

    def feature_embedding(self, img_buffer, embedding_buffer, frame_index):
        # minimal temporal length for processor
        min_t = (self.EmbedWindowLength - 1) * self.EmbedWindowAtrous

        if frame_index > min_t:
            num_embedding = embedding_buffer.shape[-1]
            img_buffer = list(img_buffer)
            curr_t = self.EmbedWindowStride * num_embedding + (self.EmbedWindowLength - 1) * self.EmbedWindowAtrous
            while curr_t < frame_index:
                # absolute index in temporal shaft
                start_index = self.EmbedWindowStride * num_embedding

                if frame_index > len(img_buffer):
                    # absolute index in buffer shaft
                    start_index = start_index - (frame_index - len(img_buffer))

                input_data = [
                    [cv2.resize(img_buffer[start_index + i * self.EmbedWindowAtrous],
                                (self.ImgSizeHeight, self.ImgSizeWidth)) for i in range(self.EmbedWindowLength)]
                    for j in range(self.EmbedBatchSize)]

                out_logits = self.encoder.infer(
                    inputs={self.encoder_input_keys[0]: input_data})[self.encoder_output_key[0]]
                out_logits = out_logits.squeeze((0, 3, 4))
                embedding_buffer = np.concatenate([embedding_buffer, out_logits],
                                                  axis = 1)  # ndarray: C x num_embedding

                curr_t += self.EmbedWindowStride

        return embedding_buffer

    def action_segmentation(self):
        # read buffer
        embed_buffer_top = self.EmbedBufferTop
        embed_buffer_side = self.EmbedBufferSide
        batch_size = self.SegBatchSize
        start_index = self.TemporalLogits.shape[0]
        end_index = min(embed_buffer_top.shape[-1], embed_buffer_side.shape[-1])
        num_batch = (end_index - start_index) // batch_size
        if num_batch < 0:
            log.debug("Waiting for the next frame ...")
        elif num_batch == 0:
            log.debug(f"start_index: {start_index} end_index: {end_index}")

            unit1 = embed_buffer_top[:, start_index:end_index]
            unit2 = embed_buffer_side[:, start_index:end_index]
            feature_unit = np.concatenate([unit1[:, ], unit2[:, ]], axis=0)
            input_mstcn = np.expand_dims(feature_unit, 0)

            feed_dict = {}
            if len(self.his_fea) != 0:
                feed_dict = {self.mstcn_input_keys[i]: self.his_fea[i] for i in range(4)}
            feed_dict[self.mstcn_input_keys[-1]] = input_mstcn
            if input_mstcn.shape == (1, 2560, 1):
                out = self.reshape_mstcn.infer(inputs=feed_dict)

            predictions = out[self.mstcn_output_key[-1]]
            self.his_fea = [out[self.mstcn_output_key[i]] for i in range(4)]

            temporal_logits = predictions[:, :, :len(self.ActionTerms), :] # 4x1x16xN
            temporal_logits = softmax(temporal_logits[-1], 1)  # 1x16xN
            temporal_logits = temporal_logits.transpose((0, 2, 1)).squeeze(axis=0)
            self.TemporalLogits = np.concatenate([self.TemporalLogits, temporal_logits], axis = 0)
        else:
            for batch_idx in range(num_batch):
                unit1 = embed_buffer_top[: ,
                        start_index + batch_idx * batch_size:start_index + batch_idx * batch_size + batch_size]
                unit2 = embed_buffer_side[: ,
                        start_index + batch_idx * batch_size:start_index + batch_idx * batch_size + batch_size]
                feature_unit = np.concatenate([unit1[:, ], unit2[:, ]], axis = 0)
                feed_dict = {}
                if len(self.his_fea) != 0:
                    feed_dict = {self.mstcn_input_keys[i]: self.his_fea[i] for i in range(4)}
                feed_dict[self.mstcn_input_keys[-1]] = feature_unit
                out = self.mstcn.infer(inputs=feed_dict)
                predictions = out[self.mstcn_output_key[-1]]
                self.his_fea = [out[self.mstcn_output_key[i]] for i in range(4)]

                temporal_logits = predictions[:, :, :len(self.ActionTerms), :] # 4x1x16xN
                temporal_logits = softmax(temporal_logits[-1], 1)  # 1x16xN
                temporal_logits = temporal_logits.transpose((0, 2, 1)).squeeze(axis = 0)
                self.TemporalLogits = np.concatenate([self.TemporalLogits, temporal_logits], axis = 0)
