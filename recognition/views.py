
import cv2
import numpy as np
import mediapipe as mp
from django.http import StreamingHttpResponse
from django.shortcuts import render

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)

# Function to interpret hand gestures
def recognize_gesture(landmarks):
    """
    Determines the hand gesture based on the position of the fingers.
    Returns a string representing the recognized gesture.
    """
    # Finger tip indexes (thumb, index, middle, ring, pinky)
    tip_ids = [4, 8, 12, 16, 20]

    fingers = []
    for i in range(1, 5):  # Loop over index to pinky
        if landmarks[tip_ids[i]].y < landmarks[tip_ids[i] - 2].y:  
            fingers.append(1)  # Finger is extended
        else:
            fingers.append(0)  # Finger is closed

    # Thumb (sideways detection)
    if landmarks[tip_ids[0]].x > landmarks[tip_ids[0] - 1].x:  
        fingers.insert(0, 1)  # Thumb is open
    else:
        fingers.insert(0, 0)  # Thumb is closed

    # Gesture recognition based on finger positions
    
    if fingers == [0, 1, 0, 0, 0]:  
        return "Letter L"
    elif fingers == [1, 0, 0, 0, 0]:
        return "Letter A"
    elif fingers == [0, 1, 1, 0, 0]:  
        return "Letter U"
    elif fingers == [1, 1, 1, 1, 1]:  
        return "Open Hand"
    elif fingers == [0, 0, 0, 0, 0]:  
        return "Fist"
    elif fingers == [1, 0, 0, 0, 1]:  
        return "Rock Sign 🤘"
    else:
        return "Unknown Gesture"

# Function to process video frames
def generate_frames():
    cap = cv2.VideoCapture(0)  # Open webcam

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)  # Mirror effect
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        gesture = "No Hand Detected"
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Get hand landmark positions
                h, w, _ = frame.shape
                landmarks = []
                for lm in hand_landmarks.landmark:
                    landmarks.append(lm)

                # Recognize gesture
                gesture = recognize_gesture(landmarks)

        # Display the detected gesture
        cv2.putText(frame, f"Gesture: {gesture}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                    1, (0, 255, 0), 2, cv2.LINE_AA)

        # Convert frame to JPEG
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

def video_feed(request):
    return StreamingHttpResponse(generate_frames(), content_type='multipart/x-mixed-replace; boundary=frame')

def index(request):
    return render(request, 'recognition/index.html')