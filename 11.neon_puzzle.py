# 사용 방법:
# 1. 양손 엄지 끝으로 사각형을 만들어 3초간 유지하면 퍼즐이 생성됩니다.
# 2. 엄지와 검지를 맞대어 조각을 잡고 원하는 위치로 옮겨 사진을 완성하세요.
# 3. 우측 상단 RESET 버튼을 엄지와 검지로 집으면 다시 시작할 수 있습니다.

import cv2
import mediapipe as mp
import time
import numpy as np
import random

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

NEON_GREEN_GLOW = (57, 255, 20)
NEON_GREEN_CORE = (100, 255, 100)

hands = mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.8)

cap = cv2.VideoCapture(0)
w_val, h_val = 1280, 720
cap.set(cv2.CAP_PROP_FRAME_WIDTH, w_val)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h_val)

is_frozen = False
puzzle_complete = False
flash_timer = 0
puzzle_rect = None  
pieces, piece_order = [], []
selected_idx, start_grid_idx = -1, -1
drag_pos = [0, 0]   
last_pinch_point = None 
start_time = None

btn_x1, btn_y1, btn_x2, btn_y2 = 1000, 50, 1200, 130

def get_dist(p1, p2):
    return np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

def draw_neon_btn(image, x1, y1, x2, y2, text, active=False):
    color_glow = NEON_GREEN_GLOW if active else (230, 230, 230)
    color_core = NEON_GREEN_CORE if active else (255, 255, 255)
    cv2.rectangle(image, (x1, y1), (x2, y2), color_glow, 5, cv2.LINE_AA)
    cv2.rectangle(image, (x1, y1), (x2, y2), color_core, 1, cv2.LINE_AA)
    cv2.putText(image, text, (x1+35, y1+55), cv2.FONT_HERSHEY_DUPLEX, 1.2, color_core, 2, cv2.LINE_AA)

def draw_neon_glow_hands(image, hand_landmarks_list):
    if not hand_landmarks_list: return
    glow_mask = np.zeros_like(image)
    for hand_landmarks in hand_landmarks_list:
        mp_drawing.draw_landmarks(glow_mask, hand_landmarks, mp_hands.HAND_CONNECTIONS, 
                                  None, mp_drawing.DrawingSpec(color=NEON_GREEN_GLOW, thickness=12))
    glow_blur = cv2.GaussianBlur(glow_mask, (25, 25), 0)
    cv2.addWeighted(image, 1.0, glow_blur, 0.7, 0, image)
    for hand_landmarks in hand_landmarks_list:
        mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS, 
                                  mp_drawing.DrawingSpec(color=(0,0,0), thickness=0, circle_radius=0), 
                                  mp_drawing.DrawingSpec(color=NEON_GREEN_CORE, thickness=2))

def reset_game():
    global is_frozen, puzzle_complete, selected_idx, start_grid_idx, start_time, pieces, puzzle_rect
    is_frozen, puzzle_complete = False, False
    selected_idx, start_grid_idx = -1, -1
    start_time, pieces, puzzle_rect = None, [], None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    frame = cv2.flip(frame, 1)
    display_frame = frame.copy()
    results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    pinch_point, is_pinching = None, False
    thumb_points = []
    hand_in_btn = False

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            t_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            i_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            tx, ty = int(t_tip.x * w_val), int(t_tip.y * h_val)
            ix, iy = int(i_tip.x * w_val), int(i_tip.y * h_val)
            thumb_points.append((tx, ty))
            
            mid_x, mid_y = (tx + ix) // 2, (ty + iy) // 2
            if btn_x1 < mid_x < btn_x2 and btn_y1 < mid_y < btn_y2:
                hand_in_btn = True

            if get_dist((tx, ty), (ix, iy)) < 40:
                is_pinching = True
                pinch_point = (mid_x, mid_y)
                last_pinch_point = pinch_point

    if is_frozen and puzzle_rect:
        x1, y1, x2, y2 = puzzle_rect
        pw, ph = (x2-x1)//3, (y2-y1)//3
        for grid_idx in range(9):
            if grid_idx == start_grid_idx: continue
            r, c = grid_idx // 3, grid_idx % 3
            display_frame[y1+r*ph:y1+(r+1)*ph, x1+c*pw:x1+(c+1)*pw] = pieces[piece_order[grid_idx]]

        if not puzzle_complete:
            if is_pinching and pinch_point:
                cx, cy = pinch_point
                if selected_idx == -1 and x1 < cx < x2 and y1 < cy < y2:
                    col, row = (cx - x1) // pw, (cy - y1) // ph
                    start_grid_idx = int(row * 3 + col)
                    selected_idx = piece_order[start_grid_idx]
                elif selected_idx != -1: drag_pos = [cx - pw//2, cy - ph//2]
            elif selected_idx != -1:
                if last_pinch_point:
                    lx, ly = last_pinch_point
                    if x1 < lx < x2 and y1 < ly < y2:
                        tc, tr = (lx - x1) // pw, (ly - y1) // ph
                        target = int(tr * 3 + tc)
                        piece_order[start_grid_idx], piece_order[target] = piece_order[target], piece_order[start_grid_idx]
                selected_idx, start_grid_idx = -1, -1
                if piece_order == list(range(9)): puzzle_complete = True

        if selected_idx != -1:
            dx, dy = max(0, min(w_val-pw, drag_pos[0])), max(0, min(h_val-ph, drag_pos[1]))
            display_frame[dy:dy+ph, dx:dx+pw] = pieces[selected_idx]
            cv2.rectangle(display_frame, (dx, dy), (dx+pw, dy+ph), NEON_GREEN_GLOW, 5, cv2.LINE_AA)

    if puzzle_complete:
        draw_neon_btn(display_frame, btn_x1, btn_y1, btn_x2, btn_y2, "RESET", active=hand_in_btn)
        cv2.putText(display_frame, "COMPLETE!", (puzzle_rect[0], puzzle_rect[1]-20), cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 0), 4, cv2.LINE_AA)
        if is_pinching and pinch_point:
            if btn_x1 < pinch_point[0] < btn_x2 and btn_y1 < pinch_point[1] < btn_y2:
                reset_game()

    if results.multi_hand_landmarks:
        draw_neon_glow_hands(display_frame, results.multi_hand_landmarks)

    if not is_frozen:
        if len(thumb_points) == 2:
            p1, p2 = thumb_points[0], thumb_points[1]
            rx1, rx2, ry1, ry2 = min(p1[0], p2[0]), max(p1[0], p2[0]), min(p1[1], p2[1]), max(p1[1], p2[1])
            if (rx2-rx1) > 150 and (ry2-ry1) > 150:
                cv2.rectangle(display_frame, (rx1, ry1), (rx2, ry2), (255, 255, 255), 2, cv2.LINE_AA)
                if start_time is None: start_time = time.time()
                countdown = 3 - int(time.time() - start_time)
                if countdown > 0:
                    text = f"READY {countdown}"
                    pos = (rx1 + (rx2-rx1)//2 - 120, ry1 - 30)
                    cv2.putText(display_frame, text, pos, cv2.FONT_HERSHEY_DUPLEX, 1.8, (0, 0, 0), 8, cv2.LINE_AA)
                    cv2.putText(display_frame, text, pos, cv2.FONT_HERSHEY_DUPLEX, 1.8, (255, 255, 255), 3, cv2.LINE_AA)
                else:
                    puzzle_rect = (rx1, ry1, rx2, ry2)
                    roi = frame[ry1:ry2, rx1:rx2]
                    ph, pw = (ry2-ry1)//3, (rx2-rx1)//3
                    pieces = [cv2.resize(roi[i*ph:(i+1)*ph, j*pw:(j+1)*pw].copy(), (pw, ph)) for i in range(3) for j in range(3)]
                    piece_order = list(range(9))
                    random.shuffle(piece_order)
                    is_frozen, flash_timer = True, 5
        else: start_time = None

    if flash_timer > 0:
        display_frame = cv2.addWeighted(display_frame, 0.4, np.full(display_frame.shape, 255, dtype=np.uint8), 0.6, 0)
        flash_timer -= 1

    cv2.imshow('Neon Puzzle UI Improved', display_frame)
    if cv2.waitKey(1) == ord('q'): break
    elif cv2.waitKey(1) == ord('r'): reset_game()

cap.release()
cv2.destroyAllWindows()