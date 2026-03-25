'''
1. 그리기 시작
 - 양손의 엄지와 검지 끝으로 사각형 틀을 만듭니다. (ㄴㄱ 모양.)
 - 틀 안의 영역이 유리 효과로 변환됩니다.
2. 효과 확인
 - 움직이는 틀 안의 배경이 흐리고 반투명한 유리 질감으로 표시됩니다.
'''
import cv2
import mediapipe as mp
import numpy as np

def apply_ios_glass_effect(image, x1, y1, x2, y2):
    sh, sw, _ = image.shape
    x1, x2 = max(0, min(x1, x2)), min(sw, max(x1, x2))
    y1, y2 = max(0, min(y1, y2)), min(sh, max(y1, y2))
    if x2 - x1 < 80 or y2 - y1 < 80: return image
    roi = image[y1:y2, x1:x2].copy()
    rh, rw, _ = roi.shape
    blurred_roi = cv2.GaussianBlur(roi, (91, 91), 0)
    noise = np.zeros((rh, rw, 3), dtype=np.uint8)
    cv2.randn(noise, (0, 0, 0), (10, 10, 10))
    glass_with_noise = cv2.add(blurred_roi, noise)
    overlay = np.full((rh, rw, 3), 255, dtype=np.uint8)
    ios_glass_roi = cv2.addWeighted(glass_with_noise, 0.80, overlay, 0.20, 0)
    result = image.copy()
    result[y1:y2, x1:x2] = ios_glass_roi
    cv2.rectangle(result, (x1, y1), (x2, y2), (240, 240, 240), 1)
    return result

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.8,
    min_tracking_confidence=0.8
)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

print("시작: 양손의 엄지와 검지로 사각형 틀을 만들어보세요! (종료: q)")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)
    points = []
    if results.multi_hand_landmarks and len(results.multi_hand_landmarks) == 2:
        for hand_landmarks in results.multi_hand_landmarks:
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            points.append((int(thumb_tip.x * w), int(thumb_tip.y * h)))
            points.append((int(index_tip.x * w), int(index_tip.y * h)))
        if len(points) == 4:
            all_x = [p[0] for p in points]
            all_y = [p[1] for p in points]
            x_min, x_max = min(all_x), max(all_x)
            y_min, y_max = min(all_y), max(all_y)
            frame = apply_ios_glass_effect(frame, x_min, y_min, x_max, y_max)
    cv2.imshow('iOS Style Glassmorphism Framework', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
hands.close()
