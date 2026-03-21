'''
1. 웹캠을 이용해 얼굴의 눈 움직임을 인식하고, 눈을 깜빡일 때마다 횟수를 자동으로 세어주는 프로그램입니다.
2. MediaPipe와 OpenCV를 활용해 실시간으로 눈의 상태를 분석합니다.
'''

import cv2
import mediapipe as mp

EYE_AR_THRESH = 0.013  
TOTAL = 0
is_open = False

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, image = cap.read()
    if not success: break

    image = cv2.flip(image, 1)
    h, w, _ = image.shape
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_image)

    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark
        
        l_up, l_down = landmarks[159], landmarks[145]
        l_ear = abs(l_up.y - l_down.y)

        r_up, r_down = landmarks[386], landmarks[374]
        r_ear = abs(r_up.y - r_down.y)

        avg_ear = (l_ear + r_ear) / 2.0

        if avg_ear > EYE_AR_THRESH:
            is_open = True 
        else:
            if is_open == True:
                TOTAL += 1
                is_open = False

        for p in [l_up, l_down, r_up, r_down]:
            cv2.circle(image, (int(p.x * w), int(p.y * h)), 2, (0, 0, 255), -1)

        cv2.putText(image, f"Blinks: {TOTAL}", (30, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        cv2.putText(image, f"Value: {avg_ear:.4f}", (30, 100), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
    cv2.imshow('Final Blink Counter', image)
    if cv2.waitKey(5) & 0xFF == 27: break

cap.release()
cv2.destroyAllWindows()
