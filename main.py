import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

cap=cv2.VideoCapture(0)
#TODO: Add required mediapipe functions

while True:
    success, frame=cap.read()
    if not success:
        break

    # Convert the frame to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Initialize the hand model
    with mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5) as hands:
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Extract landmark positions
                landmarks = np.array([[landmark.x, landmark.y, landmark.z] for landmark in hand_landmarks.landmark])

                # Check for left or right hand
                if landmarks[0, 0] < landmarks[17, 0]:
                    hand_type = "Left"
                else:
                    hand_type = "Right"

                # Draw landmarks on the frame
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                # Display the hand type
                cv2.putText(frame, hand_type, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    cv2.imshow("Hand Detection",frame)

    #TODO: Implement the functions

    if cv2.waitKey(1) & 0xFF==ord('q'):
        break
