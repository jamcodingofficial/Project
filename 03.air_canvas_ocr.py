'''
1. 프로그램 실행 시 카메라가 켜지며 실시간 영상이 나타납니다.
2. 작동 방법:
 - 선 그리는 방법: 검지만 펼친다.
 - 선 지우는 방법: 검지+중지를 펼친다.
 - 모두 지우는 방법: 엄지+검지+중지 세 손가락을 펼친다.
 - 문자 인식 방법: 다섯 손가락을 모두 펼치면 화면 중앙에 결과가 출력된다.
'''
import cv2
import mediapipe as mp
import numpy as np
import pytesseract
import time
import re

ERASER_SIZE = 25
DRAW_THICKNESS = 12

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.8,
    min_tracking_confidence=0.8
)

canvas = None
prev_x, prev_y = None, None
recognized_text = ""
display_start_time = 0
current_scale = 0.1

cap = cv2.VideoCapture(0)

def get_finger_status(hand_landmarks):
    fingers = []
    if hand_landmarks.landmark[4].x < hand_landmarks.landmark[3].x:
        fingers.append(True)
    else:
        fingers.append(False)
        
    tips = [8, 12, 16, 20] 
    pips = [6, 10, 14, 18]
    
    for tip, pip in zip(tips, pips):
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[pip].y:
            fingers.append(True)
        else:
            fingers.append(False)
    return fingers

while cap.isOpened():
    success, image = cap.read()
    if not success: break

    image = cv2.flip(image, 1)
    h, w, c = image.shape
    if canvas is None: 
        canvas = np.zeros_like(image)

    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_image)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            fingers = get_finger_status(hand_landmarks)
            idx_tip = hand_landmarks.landmark[8]
            tx, ty = int(idx_tip.x * w), int(idx_tip.y * h)

            if fingers[0] and fingers[1] and fingers[2] and not fingers[3] and not fingers[4]:
                canvas = np.zeros_like(image)
                prev_x, prev_y = None, None

            elif not fingers[0] and fingers[1] and fingers[2] and not fingers[3] and not fingers[4]:
                cv2.circle(canvas, (tx, ty), ERASER_SIZE, (0, 0, 0), -1)
                prev_x, prev_y = None, None

            elif all(fingers):
                prev_x, prev_y = None, None
                if time.time() - display_start_time > 3.0:
                    gray_canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
                    coords = cv2.findNonZero(gray_canvas)
                    if coords is not None:
                        x, y, bw, bh = cv2.boundingRect(coords)
                        pad = 30
                        roi = gray_canvas[max(0, y-pad):min(h, y+bh+pad), 
                                         max(0, x-pad):min(w, x+bw+pad)]
                        
                        roi_resized = cv2.resize(roi, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
                        _, thresh = cv2.threshold(roi_resized, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                        thresh = cv2.GaussianBlur(thresh, (3, 3), 0)

                        try:
                            config = '--psm 6 --oem 3'
                            result = pytesseract.image_to_string(thresh, lang='kor+eng', config=config).strip()
                            result = re.sub(r'[^\w\s]', '', result)
                            recognized_text = result if result else "Try again!"
                        except:
                            recognized_text = "OCR Error!"
                    else:
                        recognized_text = "Write something!"
                    
                    display_start_time = time.time()
                    current_scale = 0.1

            elif fingers[1] and not fingers[2]:
                if prev_x is not None:
                    cv2.line(canvas, (prev_x, prev_y), (tx, ty), (0, 255, 255), DRAW_THICKNESS)
                prev_x, prev_y = tx, ty
                cv2.circle(image, (tx, ty), 6, (0, 255, 0), -1)
            
            else:
                prev_x, prev_y = None, None

    combined = cv2.addWeighted(image, 0.6, canvas, 0.4, 0)

    if recognized_text and (time.time() - display_start_time < 3):
        if current_scale < 1.0: current_scale += 0.1
        text_size = cv2.getTextSize(recognized_text, cv2.FONT_HERSHEY_SIMPLEX, 1.5 * current_scale, 3)[0]
        center_x, center_y = w // 2, h // 2
        cv2.putText(combined, recognized_text, (center_x - text_size[0]//2, center_y + text_size[1]//2), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5 * current_scale, (255, 255, 255), 3, cv2.LINE_AA)

    cv2.imshow('Air Canvas Project', combined)
    if cv2.waitKey(1) & 0xFF == 27: break

cap.release()
cv2.destroyAllWindows()