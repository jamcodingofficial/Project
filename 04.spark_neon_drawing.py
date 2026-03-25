import cv2
import mediapipe as mp
import numpy as np
import math
import random

SKELETON_COLOR = (255, 253, 0)
DRAW_GLOW_COLOR = (0, 0, 255)
GRAB_GLOW_COLOR = (0, 255, 255)
SPARK_CORE_COLOR = (255, 255, 255)
SPARK_GLOW_COLOR = (0, 165, 255)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.8)

class PinchSpark:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        angle = random.uniform(0, 2 * math.pi)
        speed = random.uniform(3, 8)
        self.vx = math.cos(angle) * speed
        self.vy = math.sin(angle) * speed
        self.life = random.randint(10, 22)
        self.size = random.uniform(1.2, 2.8)

    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.life -= 1
        self.vy += 0.3

    def draw_core(self, canvas):
        if self.life <= 0: return
        alpha = self.life / 22.0
        size = int(self.size * alpha) + 1
        cv2.circle(canvas, (int(self.x), int(self.y)), size, (0, int(255 * alpha), 255), -1)

    def draw_glow(self, canvas):
        if self.life <= 0: return
        alpha = (self.life / 22.0) * 0.7
        size = int(self.size * alpha * 2) + 1
        cv2.circle(canvas, (int(self.x), int(self.y)), size, (0, int(255 * alpha), 255), -1)

all_sparks = []
shapes = []
current_shape = []
selected_shape_idx = None
prev_pinch_pos = None

def spawn_sparks(x, y, count=10):
    for _ in range(count):
        all_sparks.append(PinchSpark(x, y))

def get_dist(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

def find_closest_shape(pinch_pos, all_shapes):
    min_dist = 50
    closest_idx = None
    for i, shape in enumerate(all_shapes):
        for pt in shape:
            d = get_dist(pinch_pos, pt)
            if d < min_dist:
                min_dist = d
                closest_idx = i
    return closest_idx

cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, image = cap.read()
    if not success: break

    image = cv2.flip(image, 1)
    h, w, _ = image.shape
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    display_frame = np.zeros_like(image)
    glow_canvas = np.zeros_like(image)
    spark_core_canvas = np.zeros_like(image)
    spark_glow_canvas = np.zeros_like(image)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(display_frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=SKELETON_COLOR, thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=SKELETON_COLOR, thickness=2))

            thumb = hand_landmarks.landmark[4]
            index = hand_landmarks.landmark[8]
            middle = hand_landmarks.landmark[12]
            
            tx, ty = int(thumb.x * w), int(thumb.y * h)
            ix, iy = int(index.x * w), int(index.y * h)
            mx, my = int(middle.x * w), int(middle.y * h)
            
            pinch_center = (ix, iy)
            dist = get_dist((tx, ty), (ix, iy))

            if dist < 45:
                if current_shape:
                    shapes.append(current_shape)
                    current_shape = []
                if selected_shape_idx is None:
                    selected_shape_idx = find_closest_shape(pinch_center, shapes)
                if selected_shape_idx is not None and prev_pinch_pos:
                    dx = pinch_center[0] - prev_pinch_pos[0]
                    dy = pinch_center[1] - prev_pinch_pos[1]
                    shapes[selected_shape_idx] = [(p[0]+dx, p[1]+dy) for p in shapes[selected_shape_idx]]
                prev_pinch_pos = pinch_center
            
            else:
                selected_shape_idx = None
                prev_pinch_pos = None
                
                if iy < hand_landmarks.landmark[6].y * h and my > hand_landmarks.landmark[10].y * h:
                    current_shape.append((ix, iy))
                    if len(current_shape) % 3 == 0:
                        spawn_sparks(ix, iy, count=random.randint(5, 12)) 
                else:
                    if current_shape:
                        shapes.append(current_shape)
                        current_shape = []

    for i, shape in enumerate(shapes):
        color = GRAB_GLOW_COLOR if i == selected_shape_idx else DRAW_GLOW_COLOR
        for j in range(1, len(shape)):
            cv2.line(glow_canvas, shape[j-1], shape[j], color, 6)
            
    for j in range(1, len(current_shape)):
        cv2.line(glow_canvas, current_shape[j-1], current_shape[j], DRAW_GLOW_COLOR, 6)

    new_sparks = []
    for spark in all_sparks:
        spark.update()
        if spark.life > 0:
            spark.draw_glow(spark_glow_canvas)
            spark.draw_core(spark_core_canvas)
            new_sparks.append(spark)
    all_sparks = new_sparks

    glow = cv2.GaussianBlur(glow_canvas, (25, 25), 0)
    spark_glow_blur = cv2.GaussianBlur(spark_glow_canvas, (13, 13), 0)
    
    step1 = cv2.addWeighted(display_frame, 1.0, spark_glow_blur, 1.2, 0)
    step2 = cv2.addWeighted(step1, 1.0, spark_core_canvas, 1.0, 0)
    
    final_output = cv2.addWeighted(step2, 1.0, glow, 2.0, 0)
    
    core = np.zeros_like(glow_canvas)
    gray = cv2.cvtColor(glow_canvas, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    core[mask > 0] = (255, 255, 255)
    
    final_output = cv2.addWeighted(final_output, 1.0, core, 1.0, 0)

    cv2.imshow('Real Spark Neon Drawing', final_output)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'): break
    elif key == ord('c'): shapes = []; current_shape = []; all_sparks = []

cap.release()
cv2.destroyAllWindows()