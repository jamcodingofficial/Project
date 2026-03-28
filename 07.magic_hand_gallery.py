'''
1. 카메라로 손 제스처를 인식하여 동작에 따라 서로 다른 이미지를 보여주는 인터랙티브 프로그램입니다.
 - 한 손으로 브이 포즈
 - 양손 주먹을 쥐고 카메라를 향해서 손바닥 쪽이 보이는 포즈
 - 양손 따봉
 - 양손 펼침 
2. 간단한 손동작만으로 화면 속 결과를 바꾸며 재미있게 체험할 수 있습니다.
3. 04. img 폴더에 있는 이미지를 다운로드 받아야 합니다.
'''
import cv2
import mediapipe as mp
import numpy as np
import os
import math

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.8,
    min_tracking_confidence=0.8
)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

def load_safe(path, color):
    target_w, target_h = 640, 720
    if os.path.exists(path):
        img = cv2.imread(path)
        if img is None:
            blank = np.zeros((target_h, target_w, 3), np.uint8)
            blank[:] = color
            return blank
        return cv2.resize(img, (target_w, target_h))
    else:
        blank = np.zeros((target_h, target_w, 3), np.uint8)
        blank[:] = color
        cv2.putText(blank, f"{path} Missing", (50, 360), 1, 1.5, (255,255,255), 2)
        return blank

def get_dist(p1, p2):
    return math.hypot(p1.x - p2.x, p1.y - p2.y)

img_v_a = load_safe('04.img/a.png', (0, 150, 0))
img_fist_b = load_safe('04.img/b.png', (0, 0, 150))
img_thumb_c = load_safe('04.img/c.png', (150, 0, 0))
img_both_spread_d = load_safe('04.img/d.png', (150, 150, 0))
img_waiting = np.zeros((720, 640, 3), np.uint8)
cv2.putText(img_waiting, "Waiting...", (220, 360), 1, 1.5, (100, 100, 100), 2)

cv2.namedWindow('Camera')
cv2.moveWindow('Camera', 20, 100)
cv2.namedWindow('Photo Gallery')
cv2.moveWindow('Photo Gallery', 1000, 100)

while cap.isOpened():
    success, frame = cap.read()
    if not success: break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    is_v_pose = False
    fist_palm_count = 0
    thumb_up_count = 0
    spread_count = 0

    if results.multi_hand_landmarks and results.multi_handedness:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            lm = hand_landmarks.landmark
            label = handedness.classification[0].label
            
            tips = [8, 12, 16, 20]
            base = 0 
            
            dists = [get_dist(lm[t], lm[base]) for t in tips]
            is_fist_basic = all(d < 0.25 for d in dists)
            
            thumb_to_palm = get_dist(lm[4], lm[13])
            thumb_folded = thumb_to_palm < 0.15
            
            if label == 'Left':
                is_palm = lm[5].x > lm[17].x
            else:
                is_palm = lm[5].x < lm[17].x

            if is_fist_basic and thumb_folded and is_palm:
                fist_palm_count += 1
            
            elif lm[4].y < lm[3].y < lm[2].y < lm[5].y and is_fist_basic:
                thumb_up_count += 1
            
            elif all(get_dist(lm[t], lm[base]) > 0.4 for t in tips):
                spread_count += 1
            
            elif get_dist(lm[8], lm[base]) > 0.4 and get_dist(lm[12], lm[base]) > 0.4 and \
                 get_dist(lm[16], lm[base]) < 0.25 and get_dist(lm[20], lm[base]) < 0.25:
                is_v_pose = True

    current_right = img_waiting

    if fist_palm_count >= 2:
        current_right = img_fist_b
        cv2.rectangle(frame, (0, h-60), (450, h), (0, 0, 255), -1)
        cv2.putText(frame, "PALM FIST!", (10, h-20), 1, 1.8, (255, 255, 255), 2)
    elif thumb_up_count >= 2:
        current_right = img_thumb_c
        cv2.rectangle(frame, (0, h-60), (480, h), (255, 0, 0), -1)
        cv2.putText(frame, "DUAL THUMBS UP!!", (10, h-20), 1, 1.8, (255, 255, 255), 2)
    elif spread_count >= 2:
        current_right = img_both_spread_d
        cv2.rectangle(frame, (0, h-60), (480, h), (150, 150, 0), -1)
        cv2.putText(frame, "BOTH SPREAD!", (10, h-20), 1, 1.8, (255, 255, 255), 2)
    elif is_v_pose:
        current_right = img_v_a
        cv2.rectangle(frame, (0, h-60), (250, h), (0, 255, 0), -1)
        cv2.putText(frame, "V POSE!", (10, h-20), 1, 1.8, (255, 255, 255), 2)

    cv2.imshow('Camera', cv2.resize(frame, (w//2, h//2)))
    cv2.imshow('Photo Gallery', current_right)

    if cv2.waitKey(1) & 0xFF == 27: break

cap.release()
cv2.destroyAllWindows()
