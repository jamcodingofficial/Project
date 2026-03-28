'''
1. 카메라를 통해 손을 인식하고, 엄지와 검지 사이의 거리를 이용해 컴퓨터의 볼륨을 직관적으로 조절할 수 있는 프로그램입니다.
 - 엄지와 검지가 맞닿아 있으면(ok사인) 컴퓨터 볼륨이 0입니다. 엄지와 검지가 점점 멀어질수록 컴퓨터 볼륨이 증가합니다.
2. 손가락을 벌리거나 오므리는 간단한 동작만으로 실시간 볼륨 제어가 가능합니다..
3. 유튜브에서 노래를 재생한 상태에서 손가락으로 볼륨을 제어하는 것을 보여주면 좋습니다.
'''
import cv2
import mediapipe as mp
import numpy as np
import math
import os

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

prev_vol = -1
vol_per = 0
vol_bar = 400

while cap.isOpened():
    success, img = cap.read()
    if not success: break

    img = cv2.flip(img, 1)
    h, w, c = img.shape
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_img)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(img, handLms, mp_hands.HAND_CONNECTIONS)

            lmList = handLms.landmark
            x1, y1 = int(lmList[4].x * w), int(lmList[4].y * h)
            x2, y2 = int(lmList[8].x * w), int(lmList[8].y * h)
            
            length = math.hypot(x2 - x1, y2 - y1)

            vol_per = int(np.interp(length, [50, 250], [0, 100]))
            vol_bar = np.interp(length, [50, 250], [400, 150])

            if abs(vol_per - prev_vol) > 2:
                os.system(f"osascript -e 'set volume output volume {vol_per}'")
                prev_vol = vol_per

            cv2.circle(img, (x1, y1), 10, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 10, (255, 0, 255), cv2.FILLED)
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

    cv2.rectangle(img, (50, 150), (85, 400), (255, 0, 0), 3)
    cv2.rectangle(img, (50, int(vol_bar)), (85, 400), (255, 0, 0), cv2.FILLED)
    cv2.putText(img, f'{int(vol_per)} %', (40, 450), 
                cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 3)

    cv2.imshow("Optimized Mac Vol Control", img)
    if cv2.waitKey(1) & 0xFF == 27: break

cap.release()
cv2.destroyAllWindows()
