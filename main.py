#!/usr/bin/env python
# -*- coding: utf-8 -*-
import copy
import math
import cv2 as cv
import numpy as np
import mediapipe as mp
import streamlit as st
from utils import CvFpsCalc
from PIL import Image

def main():
    st.set_page_config(layout='wide')
    st.title("테스트 픽토그램")
    st.video('video.mp4')

    col1, col2 = st.columns(2)
    col1.write("Tokyo2020 Debug")
    debug_image01_placeholder = col1.empty()
    col2.write("Tokyo2020 Pictogram")
    debug_image02_placeholder = col2.empty()

    cap_device = 0
    cap_width = 640
    cap_height = 360
    static_image_mode = False
    model_complexity = 1
    min_detection_confidence = 0.5
    min_tracking_confidence = 0.5
    rev_color = False

    # 카메라 비활성화
    cap = cv.VideoCapture(cap_device)
    if not cap.isOpened():
        st.error("카메라를 열 수 없습니다. 다른 장치나 환경에서 실행 중일 수 있습니다.")
        return

    cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=static_image_mode,
        model_complexity=model_complexity,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )

    cvFpsCalc = CvFpsCalc(buffer_len=10)

    if rev_color:
        color = (255, 255, 255)
        bg_color = (100, 33, 3)
    else:
        color = (100, 33, 3)
        bg_color = (255, 255, 255)

    while cap.isOpened():
        display_fps = cvFpsCalc.get()

        ret, image = cap.read()
        if not ret:
            break
        image = cv.flip(image, 1)
        image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        debug_image01 = copy.deepcopy(image_rgb)
        debug_image02 = np.zeros((image_rgb.shape[0], image_rgb.shape[1], 3), np.uint8)
        cv.rectangle(debug_image02, (0, 0), (image_rgb.shape[1], image_rgb.shape[0]),
                     bg_color, thickness=-1)

        results = pose.process(image_rgb)

        if results.pose_landmarks is not None:
            mp.solutions.drawing_utils.draw_landmarks(
                debug_image01, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                mp.solutions.drawing_utils.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                mp.solutions.drawing_utils.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)
            )

            color_rgb = (color[2], color[1], color[0])
            bg_color_rgb = (bg_color[2], bg_color[1], bg_color[0])
            debug_image02 = draw_stick_figure(
                debug_image02,
                results.pose_landmarks,
                color=color_rgb,
                bg_color=bg_color_rgb,
            )

        cv.putText(debug_image01, "FPS:" + str(display_fps), (10, 30),
                   cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv.LINE_AA)
        cv.putText(debug_image02, "FPS:" + str(display_fps), (10, 30),
                   cv.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv.LINE_AA)

        debug_image01_pil = Image.fromarray(debug_image01)
        debug_image02_pil = Image.fromarray(debug_image02)

        debug_image01_placeholder.image(debug_image01_pil, use_column_width=True)
        debug_image02_placeholder.image(debug_image02_pil, use_column_width=True)

        if st.button("Stop"):
            break

    cap.release()

def draw_stick_figure(
        image,
        landmarks,
        color=(100, 33, 3),
        bg_color=(255, 255, 255),
        visibility_th=0.5,
):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_point = []
    for index, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_z = landmark.z
        landmark_point.append(
            [index, landmark.visibility, (landmark_x, landmark_y), landmark_z])

    right_leg = landmark_point[23]
    left_leg = landmark_point[24]
    leg_x = int((right_leg[2][0] + left_leg[2][0]) / 2)
    leg_y = int((right_leg[2][1] + left_leg[2][1]) / 2)

    landmark_point[23][2] = (leg_x, leg_y)
    landmark_point[24][2] = (leg_x, leg_y)

    sorted_landmark_point = sorted(landmark_point,
                                   reverse=True,
                                   key=lambda x: x[3])

    (face_x, face_y), face_radius = min_enclosing_face_circle(landmark_point)

    face_x = int(face_x)
    face_y = int(face_y)
    face_radius = int(face_radius * 1.5)

    stick_radius01 = int(face_radius * (4 / 5))
    stick_radius02 = int(stick_radius01 * (3 / 4))
    stick_radius03 = int(stick_radius02 * (3 / 4))

    draw_list = [
        11,  # 오른팔
        12,  # 왼팔
        23,  # 오른다리
        24,  # 왼다리
    ]

    cv.rectangle(image, (0, 0), (image_width, image_height),
                 bg_color,
                 thickness=-1)

    cv.circle(image, (face_x, face_y), face_radius, color, -1)

    for landmark_info in sorted_landmark_point:
        index = landmark_info[0]

        if index in draw_list:
            point01 = [p for p in landmark_point if p[0] == index][0]
            point02 = [p for p in landmark_point if p[0] == (index + 2)][0]
            point03 = [p for p in landmark_point if p[0] == (index + 4)][0]

            if point01[1] > visibility_th and point02[1] > visibility_th:
                image = draw_stick(
                    image,
                    point01[2],
                    stick_radius01,
                    point02[2],
                    stick_radius02,
                    color=color,
                    bg_color=bg_color,
                )
            if point02[1] > visibility_th and point03[1] > visibility_th:
                image = draw_stick(
                    image,
                    point02[2],
                    stick_radius02,
                    point03[2],
                    stick_radius03,
                    color=color,
                    bg_color=bg_color,
                )

    return image


def min_enclosing_face_circle(landmark_point):
    landmark_array = np.empty((0, 2), int)

    index_list = [1, 4, 7, 8, 9, 10]
    for index in index_list:
        np_landmark_point = [
            np.array(
                (landmark_point[index][2][0], landmark_point[index][2][1]))
        ]
        landmark_array = np.append(landmark_array, np_landmark_point, axis=0)

    center, radius = cv.minEnclosingCircle(points=landmark_array)

    return center, radius


def draw_stick(
        image,
        point01,
        point01_radius,
        point02,
        point02_radius,
        color=(100, 33, 3),
        bg_color=(255, 255, 255),
):
    cv.circle(image, point01, point01_radius, color, -1)
    cv.circle(image, point02, point02_radius, color, -1)

    draw_list = []
    for index in range(2):
        rad = math.atan2(point02[1] - point01[1], point02[0] - point01[0])

        rad = rad + (math.pi / 2) + (math.pi * index)
        point_x = int(point01_radius * math.cos(rad)) + point01[0]
        point_y = int(point01_radius * math.sin(rad)) + point01[1]

        draw_list.append([point_x, point_y])

        point_x = int(point02_radius * math.cos(rad)) + point02[0]
        point_y = int(point02_radius * math.sin(rad)) + point02[1]

        draw_list.append([point_x, point_y])

    points = np.array((draw_list[0], draw_list[1], draw_list[3], draw_list[2]))
    cv.fillConvexPoly(image, points=points, color=color)

    return image


if __name__ == '__main__':
    main()
