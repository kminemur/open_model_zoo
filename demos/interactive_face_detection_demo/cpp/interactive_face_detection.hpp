// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <utils/default_flags.hpp>

#include <gflags/gflags.h>
#include <iostream>
#include <limits>

DEFINE_INPUT_FLAGS
DEFINE_OUTPUT_FLAGS

constexpr char help_message[] = "Print a usage message";
DEFINE_bool(h, false, help_message);

constexpr char face_detection_model_message[] = "Required. Path to an .xml file with a trained Face Detection model.";
DEFINE_string(m, "", face_detection_model_message);

constexpr char age_gender_model_message[] = "Optional. Path to an .xml file with a trained Age/Gender Recognition model.";
DEFINE_string(m_ag, "", age_gender_model_message);

constexpr char head_pose_model_message[] = "Optional. Path to an .xml file with a trained Head Pose Estimation model.";
DEFINE_string(m_hp, "", head_pose_model_message);

constexpr char emotions_model_message[] = "Optional. Path to an .xml file with a trained Emotions Recognition model.";
DEFINE_string(m_em, "", emotions_model_message);

constexpr char facial_landmarks_model_message[] = "Optional. Path to an .xml file with a trained Facial Landmarks Estimation model.";
DEFINE_string(m_lm, "", facial_landmarks_model_message);

constexpr char antispoofing_model_message[] = "Optional. Path to an .xml file with a trained Antispoofing Classification model.";
DEFINE_string(m_am, "", antispoofing_model_message);

constexpr char device_message[] =
    "Specify a target device to infer on (the list of available devices is shown below). "
    "Default value is CPU. Use \"-d HETERO:<comma-separated_devices_list>\" format to specify "
    "HETERO plugin. "
    "Use \"-d MULTI:<comma-separated_devices_list>\" format to specify MULTI plugin. "
    "The application looks for a suitable plugin for the specified device.";
DEFINE_string(d, "CPU", device_message);

constexpr char thresh_output_message[] = "Optional. Probability threshold for detections";
DEFINE_double(t, 0.5, thresh_output_message);

constexpr char bb_enlarge_coef_output_message[] = "Optional. Coefficient to enlarge/reduce the size of the bounding box around the detected face";
DEFINE_double(bb_enlarge_coef, 1.2, bb_enlarge_coef_output_message);

constexpr char raw_output_message[] = "Optional. Output inference results as raw values";
DEFINE_bool(r, false, raw_output_message);

constexpr char no_show_message[] = "Optional. Don't show output.";
DEFINE_bool(no_show, false, no_show_message);

constexpr char dx_coef_output_message[] = "Optional. Coefficient to shift the bounding box around the detected face along the Ox axis";
DEFINE_double(dx_coef, 1, dx_coef_output_message);

constexpr char dy_coef_output_message[] = "Optional. Coefficient to shift the bounding box around the detected face along the Oy axis";
DEFINE_double(dy_coef, 1, dy_coef_output_message);

constexpr char fps_output_message[] = "Optional. Maximum FPS for playing video";
DEFINE_double(fps, -std::numeric_limits<double>::infinity(), fps_output_message);

constexpr char no_smooth_output_message[] = "Optional. Do not smooth person attributes";
DEFINE_bool(no_smooth, false, no_smooth_output_message);

constexpr char no_show_emotion_bar_message[] = "Optional. Do not show emotion bar";
DEFINE_bool(no_show_emotion_bar, false, no_show_emotion_bar_message);

constexpr char utilization_monitors_message[] = "Optional. List of monitors to show initially.";
DEFINE_string(u, "", utilization_monitors_message);

static void showUsage() {
    std::cout << std::endl;
    std::cout << "interactive_face_detection_demo [OPTION]" << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << std::endl;
    std::cout << "    -h                         " << help_message << std::endl;
    std::cout << "    -i                         " << input_message << std::endl;
    std::cout << "    -loop                      " << loop_message << std::endl;
    std::cout << "    -o \"<path>\"                " << output_message << std::endl;
    std::cout << "    -limit \"<num>\"             " << limit_message << std::endl;
    std::cout << "    -m \"<path>\"                " << face_detection_model_message << std::endl;
    std::cout << "    -m_ag \"<path>\"             " << age_gender_model_message << std::endl;
    std::cout << "    -m_hp \"<path>\"             " << head_pose_model_message << std::endl;
    std::cout << "    -m_em \"<path>\"             " << emotions_model_message << std::endl;
    std::cout << "    -m_lm \"<path>\"             " << facial_landmarks_model_message << std::endl;
    std::cout << "    -m_am \"<path>\"             " << antispoofing_model_message << std::endl;
    std::cout << "    -d <device>                " << device_message << std::endl;
    std::cout << "    -no_show                   " << no_show_message << std::endl;
    std::cout << "    -r                         " << raw_output_message << std::endl;
    std::cout << "    -t                         " << thresh_output_message << std::endl;
    std::cout << "    -bb_enlarge_coef           " << bb_enlarge_coef_output_message << std::endl;
    std::cout << "    -dx_coef                   " << dx_coef_output_message << std::endl;
    std::cout << "    -dy_coef                   " << dy_coef_output_message << std::endl;
    std::cout << "    -fps                       " << fps_output_message << std::endl;
    std::cout << "    -no_smooth                 " << no_smooth_output_message << std::endl;
    std::cout << "    -no_show_emotion_bar       " << no_show_emotion_bar_message << std::endl;
    std::cout << "    -u                         " << utilization_monitors_message << std::endl;
}
