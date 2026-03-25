'''
1. 손 제스처를 이용해 공중에 글씨를 그리고, 작성한 내용을 실시간으로 문자로 변환해주는 인터랙티브 프로그램입니다.
 1.1 그림 그리는 법: 
  - 검지손가락만 펼치고 화면에 보이게 합니다. 
  - 손을 움직임녀 그림이 그려집니다.
 1.2 그림 지우는 법:
  - 검지와 중지를 사용하여 V 모양을 만듭니다.
 1.3 글자 인식하는 법:
  - 손가락 5개 모두 펼치기 
  - 잠시 기다리면 그린 내용이 문자로 변환되어 화면 중아에 표시됩니다(정확도 다소 떨어짐 주의)
2. MediaPipe와 OCR 기술을 활용해 손동작으로 그리기, 지우기, 텍스트 인식까지 모두 수행할 수 있습니다.

'''
import cv2
import mediapipe as mp
import numpy as np
import pytesseract
import time
import re

# Mediapipe 설정
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.8,
    min_tracking_confidence=0.8
)

# 전역 변수 초기화
canvas = None
prev_x, prev_y = None, None
recognized_text = ""
display_start_time = 0
current_scale = 0.1

cap = cv2.VideoCapture(0)

def get_finger_status(hand_landmarks):
    """손가락이 펴졌는지 접혔는지 판단 (검지, 중지, 약지, 소지만 체크)"""
    fingers = []
    tips = [8, 12, 16, 20]
    pips = [6, 10, 14, 18]
    
    for tip, pip in zip(tips, pips):
        # Y좌표 비교 (화면상 위로 갈수록 Y값이 작음)
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

            # 검지 끝마디 좌표
            idx_tip = hand_landmarks.landmark[8]
            tx, ty = int(idx_tip.x * w), int(idx_tip.y * h)

            # 1. 지우개 모드: 검지+중지만 펴고 나머지는 접었을 때 (V포즈)
            if fingers[0] and fingers[1] and not fingers[2] and not fingers[3]:
                canvas = np.zeros_like(image)
                prev_x, prev_y = None, None
                recognized_text = ""

            # 2. OCR 인식 모드: 주먹을 쥐었다가 쫙 폈을 때 (모든 손가락 펴기)
            elif all(fingers):
                prev_x, prev_y = None, None
                # 3초마다 한 번씩 인식 시도 (너무 자주 인식 방지)
                if time.time() - display_start_time > 3.0:
                    gray_canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
                    
                    # 글씨가 있는 영역 찾기
                    coords = cv2.findNonZero(gray_canvas)
                    if coords is not None:
                        # 글씨가 있는 부분만 잘라내기(Bounding Box)
                        x, y, bw, bh = cv2.boundingRect(coords)
                        pad = 30 # 여백 추가
                        roi = gray_canvas[max(0, y-pad):min(h, y+bh+pad), 
                                         max(0, x-pad):min(w, x+bw+pad)]
                        
                        # 인식률 향상을 위한 전처리: 2배 확대 및 이진화
                        roi_resized = cv2.resize(roi, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
                        _, thresh = cv2.threshold(roi_resized, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                        
                        # 가우시안 블러로 선을 부드럽게
                        thresh = cv2.GaussianBlur(thresh, (3, 3), 0)

                        try:
                            # Tesseract 설정: kor+eng 인식, psm 6(단일 텍스트 블록)
                            config = '--psm 6 --oem 3'
                            result = pytesseract.image_to_string(thresh, lang='kor+eng', config=config).strip()
                            
                            # 특수문자 제거
                            result = re.sub(r'[^\w\s]', '', result)
                            recognized_text = result if result else "Try again! ✍️"
                        except Exception as e:
                            recognized_text = "OCR Error!"
                    else:
                        recognized_text = "Write something!"
                    
                    display_start_time = time.time()
                    current_scale = 0.1

            # 3. 대기 모드: 주먹 쥐었을 때 (아무것도 안 함)
            elif not any(fingers):
                prev_x, prev_y = None, None
                cv2.circle(image, (tx, ty), 15, (0, 0, 255), 2)

            # 4. 그리기 모드: 그 외 (검지만 펴거나 그릴 때)
            else:
                if prev_x is not None:
                    # 두께 12로 부드러운 노란색 선 그리기
                    cv2.line(canvas, (prev_x, prev_y), (tx, ty), (0, 255, 255), 12)
                prev_x, prev_y = tx, ty
                cv2.circle(image, (tx, ty), 6, (0, 255, 0), -1)
    else:
        prev_x, prev_y = None, None

    # 원본 영상과 캔버스 합성
    combined = cv2.addWeighted(image, 0.6, canvas, 0.4, 0)

    # 인식된 텍스트 화면 중앙에 표시 (애니메이션 효과)
    if recognized_text and (time.time() - display_start_time < 3):
        if current_scale < 1.0: current_scale += 0.1
        
        text_size = cv2.getTextSize(recognized_text, cv2.FONT_HERSHEY_SIMPLEX, 1.5 * current_scale, 3)[0]
        center_x, center_y = w // 2, h // 2
        x1, y1 = center_x - text_size[0]//2 - 20, center_y - text_size[1]//2 - 30
        x2, y2 = center_x + text_size[0]//2 + 20, center_y + text_size[1]//2 + 30
        
        # 배경 상자 그리기
        overlay = combined.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 100, 0), -1)
        cv2.addWeighted(overlay, 0.7, combined, 0.3, 0, combined)
        
        # 글자 쓰기
        cv2.putText(combined, recognized_text, (center_x - text_size[0]//2, center_y + text_size[1]//2), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5 * current_scale, (255, 255, 255), 3, cv2.LINE_AA)

    cv2.imshow('Improved Air Canvas', combined)
    if cv2.waitKey(1) & 0xFF == 27: break # ESC 누르면 종료

cap.release()
cv2.destroyAllWindows()
