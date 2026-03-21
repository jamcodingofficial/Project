'''
1. 프로그램 실행 시 카메라가 켜집니다.
2. 눈 영역만 연한 초록색으로 표시되며, 배경은 그대로 유지됩니다.
3. 눈을 2초 이상 감으면 화면에 빨간색 알람과 “WAKE UP!!!” 문구가 나타나고, 음성으로 “지금 잠이 오니??”라고 알립니다.
 - 109번째 코드를 변경하면 출력되는 음성 변경이 가능합니다.
'''

import cv2
import mediapipe as mp
import time
import numpy as np
import os

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

cap = cv2.VideoCapture(0)

EYE_CLOSED_THRESHOLD = 0.0038 
WAIT_TIME = 2.0 

closed_start_time = 0
is_closed = False
alarm_on = False
last_say_time = 0

LEFT_EYE_CONTOUR = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
RIGHT_EYE_CONTOUR = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]

def draw_transparent_poly(img, points, color, alpha):
    overlay = img.copy()
    cv2.fillPoly(overlay, [points], color)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

print("졸음 방지 시스템 가동 중... 배경은 유지되고 눈에만 효과가 적용됩니다.")

while cap.isOpened():
    success, frame = cap.read()
    if not success: break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    current_eye_dist = 0

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            def get_lm_pos(lm_idx):
                lm = face_landmarks.landmark[lm_idx]
                return (int(lm.x * w), int(lm.y * h))

            left_eye_points = np.array([get_lm_pos(idx) for idx in LEFT_EYE_CONTOUR], dtype=np.int32)
            right_eye_points = np.array([get_lm_pos(idx) for idx in RIGHT_EYE_CONTOUR], dtype=np.int32)
            
            eye_tint_color = (150, 250, 0)
            alpha = 0.4
            
            draw_transparent_poly(frame, left_eye_points, eye_tint_color, alpha)
            draw_transparent_poly(frame, right_eye_points, eye_tint_color, alpha)
            
            left_top, left_bottom = 159, 145
            right_top, right_bottom = 386, 374
            
            def get_y_dist(p1_idx, p2_idx):
                p1 = face_landmarks.landmark[p1_idx]
                p2 = face_landmarks.landmark[p2_idx]
                return abs(p1.y - p2.y)

            left_dist = get_y_dist(left_top, left_bottom)
            right_dist = get_y_dist(right_top, right_bottom)
            
            current_eye_dist = (left_dist + right_dist) / 2

    if 0 < current_eye_dist < EYE_CLOSED_THRESHOLD:
        eye_status_text = "STATUS: EYE CLOSED"
        eye_status_color = (0, 0, 255)
        
        if not is_closed:
            closed_start_time = time.time()
            is_closed = True
        
        elapsed_time = time.time() - closed_start_time
        
        if elapsed_time >= WAIT_TIME:
            alarm_on = True
    else:
        eye_status_text = "STATUS: EYE OPEN"
        eye_status_color = (0, 255, 0)
        is_closed = False
        alarm_on = False

    if alarm_on:
        red_overlay = frame.copy()
        cv2.rectangle(red_overlay, (0, 0), (w, h), (0, 0, 255), -1)
        cv2.addWeighted(red_overlay, 0.4, frame, 0.6, 0, frame)
        
        cv2.putText(frame, "WAKE UP!!!", (w//2-180, h//2), 
                    cv2.FONT_HERSHEY_DUPLEX, 2, (255, 255, 255), 5)
        
        if time.time() - last_say_time > 1.5:
            os.system("say '지금 잠이 오니??' &")
            last_say_time = time.time()
    
    cv2.putText(frame, eye_status_text, (20, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, eye_status_color, 2)

    cv2.imshow('Drowsiness Detection with Precise Eye Tint', frame)
    
    if cv2.waitKey(1) & 0xFF == 27: break

cap.release()
cv2.destroyAllWindows()
