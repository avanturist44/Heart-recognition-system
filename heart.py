import cv2
import mediapipe as mp
import numpy as np
import glob

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

animation_frames = [cv2.imread(file, cv2.IMREAD_UNCHANGED) for file in sorted(glob.glob("heart_animation/*.png"))]

animation_playing = False
animation_index = 0
animation_position = (200, 200)  

def calculate_features(hand1, hand2):
    if hand1 is None or hand2 is None:
        return None
    
    left_thumb_tip = np.array([hand1.landmark[mp_hands.HandLandmark.THUMB_TIP].x,
                               hand1.landmark[mp_hands.HandLandmark.THUMB_TIP].y])
    left_index_tip = np.array([hand1.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x,
                               hand1.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y])
    right_thumb_tip = np.array([hand2.landmark[mp_hands.HandLandmark.THUMB_TIP].x,
                                hand2.landmark[mp_hands.HandLandmark.THUMB_TIP].y])
    right_index_tip = np.array([hand2.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x,
                                hand2.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y])
    left_wrist = np.array([hand1.landmark[mp_hands.HandLandmark.WRIST].x,
                           hand1.landmark[mp_hands.HandLandmark.WRIST].y])
    right_wrist = np.array([hand2.landmark[mp_hands.HandLandmark.WRIST].x,
                            hand2.landmark[mp_hands.HandLandmark.WRIST].y])

    thumb_tip_distance = np.linalg.norm(left_thumb_tip - right_thumb_tip)
    index_tip_distance = np.linalg.norm(left_index_tip - right_index_tip)
    left_thumb_index_distance = np.linalg.norm(left_thumb_tip - left_index_tip)
    right_thumb_index_distance = np.linalg.norm(right_thumb_tip - right_index_tip)
    wrist_distance = np.linalg.norm(left_wrist - right_wrist)

    
    return {
        "thumb_tip_distance": thumb_tip_distance,
        "index_tip_distance": index_tip_distance,
        "left_thumb_index_distance": left_thumb_index_distance,
        "right_thumb_index_distance": right_thumb_index_distance,
        "wrist_distance": wrist_distance
    }

cap = cv2.VideoCapture(1)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)
    
    if results.multi_hand_landmarks and len(results.multi_hand_landmarks) == 2:
        hand1, hand2 = results.multi_hand_landmarks[0], results.multi_hand_landmarks[1]
        
        features = calculate_features(hand1, hand2)
        
        if features:
            if (features["thumb_tip_distance"] < 0.04 and
                features["index_tip_distance"] < 0.04 and
                0.2 < features["left_thumb_index_distance"] < 0.27 and
                0.2 < features["right_thumb_index_distance"] < 0.27 and
                0.35 < features["wrist_distance"] < 0.45):
                
                animation_playing = True
                animation_index = 0  
            
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    if animation_playing:
        animation_frame = animation_frames[animation_index]
        
        h, w, _ = animation_frame.shape
        x, y = animation_position
        
        for c in range(0, 3):  
            frame[y:y+h, x:x+w, c] = np.where(
                animation_frame[:, :, 3] > 0, 
                animation_frame[:, :, c],  
                frame[y:y+h, x:x+w, c]  
            )
        
        animation_index += 1
        
        if animation_index >= len(animation_frames):
            animation_playing = False
    
    cv2.imshow('Heart Gesture Detection with Animation', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
