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

def get_args():
    """Streamlit 환경에서는 argparse 대신 Streamlit의 sidebar를 사용하여 인자를 받습니다."""
    device = st.sidebar.selectbox('Device', options=[0, 1], index=0)
    width = st.sidebar.slider('Width', 320, 1280, 640)
    height = st.sidebar.slider('Height', 240, 720, 360)
    static_image_mode = st.sidebar.checkbox('Static Image Mode', value=False)
    model_complexity = st.sidebar.selectbox('Model Complexity', options=[0, 1, 2], index=1)
    min_detection_confidence = st.sidebar.slider('Min Detection Confidence', 0.0, 1.0, 0.5)
    min_tracking_confidence = st.sidebar.slider('Min Tracking Confidence', 0.0, 1.0, 0.5)
    rev_color = st.sidebar.checkbox('Reverse Color', value=False)

    return (device, width, height, static_image_mode, model_complexity, min_detection_confidence, min_tracking_confidence, rev_color)

def main():
    # 페이지 레이아웃을 'wide'로 설정
    st.set_page_config(layout='wide')
    st.title("테스트 픽토그램")
    st.video('video.mp4')
    
    # 컬럼 생성
    col1, col2 = st.columns(2)

    # 첫 번째 컬럼에 첫 번째 이미지 표시
    col1.write("Tokyo2020 Debug")
    debug_image01_placeholder = col1.empty()

    # 두 번째 컬럼에 두 번째 이미지 표시
    col2.write("Tokyo2020 Pictogram")
    debug_image02_placeholder = col2.empty()

    # 스트림릿 사이드바에서 인자 받기
    (cap_device, cap_width, cap_height, static_image_mode, model_complexity, 
    min_detection_confidence, min_tracking_confidence, rev_color) = get_args()

    # 카메라 준비 ###############################################################
    cap = cv.VideoCapture(cap_device)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)

    # 모델 로드 ###############################################################
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=static_image_mode,
        model_complexity=model_complexity,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )

    # FPS 계산 모듈 ########################################################
    cvFpsCalc = CvFpsCalc(buffer_len=10)

    # 색 지정
    if rev_color:
        color = (255, 255, 255)
        bg_color = (100, 33, 3)
    else:
        color = (100, 33, 3)
        bg_color = (255, 255, 255)

    while True:
        display_fps = cvFpsCalc.get()

        # 카메라 캡처 #####################################################
        ret, image = cap.read()
        if not ret:
            break
        image = cv.flip(image, 1)  # 미러 표시
        image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)  # 색 공간 변경
        debug_image01 = copy.deepcopy(image_rgb)  # 변경된 색 공간으로 이미지 복사
        debug_image02 = np.zeros((image_rgb.shape[0], image_rgb.shape[1], 3), np.uint8)
        cv.rectangle(debug_image02, (0, 0), (image_rgb.shape[1], image_rgb.shape[0]),
                     bg_color,
                     thickness=-1)

        # 검출 실행 #############################################################
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        results = pose.process(image)

        # 그리기 ################################################################
        if results.pose_landmarks is not None:    
            mp.solutions.drawing_utils.draw_landmarks(
                debug_image01, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                mp.solutions.drawing_utils.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),  # 관절의 점 스타일 설정
                mp.solutions.drawing_utils.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)   # 관절을 잇는 선 스타일 설정
            )
            # 색상 순서를 RGB로 변경
            color_rgb = (color[2], color[1], color[0])
            bg_color_rgb = (bg_color[2], bg_color[1], bg_color[0])
            # 그리기
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

        # 화면 반영 #############################################################
        # 이미지를 PIL 형식으로 변환
        debug_image01_pil = Image.fromarray(debug_image01)
        debug_image02_pil = Image.fromarray(debug_image02)

        # 이미지를 스트림릿에 표시
        debug_image01_placeholder.image(debug_image01_pil, use_column_width=True)
        debug_image02_placeholder.image(debug_image02_pil, use_column_width=True)

        # 스트림릿에서 멈춤을 위한 종료 조건 설정
        if st.button('Stop'):
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

    # 각 랜드마크 계산
    landmark_point = []
    for index, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_z = landmark.z
        landmark_point.append(
            [index, landmark.visibility, (landmark_x, landmark_y), landmark_z])

    # 다리의 위치를 허리의 중점으로 수정
    right_leg = landmark_point[23]
    left_leg = landmark_point[24]
    leg_x = int((right_leg[2][0] + left_leg[2][0]) / 2)
    leg_y = int((right_leg[2][1] + left_leg[2][1]) / 2)

    landmark_point[23][2] = (leg_x, leg_y)
    landmark_point[24][2] = (leg_x, leg_y)

    # 거리 순으로 정렬
    sorted_landmark_point = sorted(landmark_point,
                                   reverse=True,
                                   key=lambda x: x[3])

    # 각 사이즈 계산
    (face_x, face_y), face_radius = min_enclosing_face_circle(landmark_point)

    face_x = int(face_x)
    face_y = int(face_y)
    face_radius = int(face_radius * 1.5)

    stick_radius01 = int(face_radius * (4 / 5))
    stick_radius02 = int(stick_radius01 * (3 / 4))
    stick_radius03 = int(stick_radius02 * (3 / 4))

    # 그리기 대상 리스트
    draw_list = [
        11,  # 오른팔
        12,  # 왼팔
        23,  # 오른다리
        24,  # 왼다리
    ]

    # 배경색
    cv.rectangle(image, (0, 0), (image_width, image_height),
                 bg_color,
                 thickness=-1)

    # 얼굴 그리기
    cv.circle(image, (face_x, face_y), face_radius, color, -1)

    # 팔/다리 그리기
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

    center, radius = cv.minEnclosingCircle(
