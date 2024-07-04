from django.shortcuts import render
from django.http import StreamingHttpResponse
import cv2
import mediapipe as mp
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
import pyautogui
import time
import pickle
import numpy as np
import pandas as pd
import joblib

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

# with open('comp.pkl', 'rb') as f:
#     model = pickle.load(f)

model = joblib.load('comp.joblib')

def home(request):
    return render(request, 'landmark/home.html')

def decreaseVolume():
    try:
        # Get the default audio device
        devices = AudioUtilities.GetSpeakers()
        interface = devices.Activate(
            IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
        volume = cast(interface, POINTER(IAudioEndpointVolume))
    
        # # Get current volume level (scalar)
        # current_volume = volume.GetMasterVolumeLevelScalar()
        # print(f"Current volume level (scalar): {current_volume * 100:.2f}%")
    
        # # Decrease volume by a specific amount (e.g., 10%)
        # decrease_amount = 0.1  # 10%
        # new_volume = max(current_volume - decrease_amount, 0.0)
        # volume.SetMasterVolumeLevelScalar(new_volume, None)
        # print(f"New volume level (scalar): {new_volume * 100:.2f}%")
    
        # Simulate a volume key press to show the volume icon
        pyautogui.press("volumedown")
        #time.sleep(0.1)  # Small delay to ensure the icon appears
    
    except Exception as e:
        print(f"An error occurred: {e}")

def increaseVolume():
    try:
        # Get the default audio device
        devices = AudioUtilities.GetSpeakers()
        interface = devices.Activate(
            IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
        volume = cast(interface, POINTER(IAudioEndpointVolume))
    
        # # Get current volume level (scalar)
        # current_volume = volume.GetMasterVolumeLevelScalar()
        # print(f"Current volume level (scalar): {current_volume * 100:.2f}%")
    
        # # Decrease volume by a specific amount (e.g., 10%)
        # decrease_amount = 0.1  # 10%
        # new_volume = max(current_volume - decrease_amount, 0.0)
        # volume.SetMasterVolumeLevelScalar(new_volume, None)
        # print(f"New volume level (scalar): {new_volume * 100:.2f}%")
    
        # Simulate a volume key press to show the volume icon
        pyautogui.press("volumeup")
        #time.sleep(0.1)  # Small delay to ensure the icon appears
    
    except Exception as e:
        print(f"An error occurred: {e}")

def gen_frames():  # generate frame by frame from camera
    cap = cv2.VideoCapture(0)
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            else:
                # Recolor Feed
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False        

                # Make Detections
                results = holistic.process(image)
                
                # Recolor image back to BGR for rendering
                image.flags.writeable = True   
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                # 1. Draw face landmarks (commented out)
                # mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION, 
                #                          mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
                #                          mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1))

                # 2. Right hand
                mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                          mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),
                                          mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2))

                # 3. Left Hand
                mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                          mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
                                          mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2))

                # 4. Pose Detections (commented out)
                # mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS, 
                #                          mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
                #                          mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2))

                body_language_class = ''
                if results.right_hand_landmarks:
                    pose = results.right_hand_landmarks.landmark
                    pose_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in pose]).flatten())
                    
                    row = pose_row

                    # Make Detections
                    X = pd.DataFrame([row])
                    body_language_class = model.predict(X)[0]
                    body_language_prob = model.predict_proba(X)[0]
                    # print(body_language_class, body_language_prob)
                    
                    # Grab ear coords
                    coords = tuple(np.multiply(
                                    np.array(
                                        (results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EAR].x, 
                                         results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EAR].y))
                                , [640,480]).astype(int))
                    
                    cv2.rectangle(image, 
                                  (coords[0], coords[1]+5), 
                                  (coords[0]+len(body_language_class)*20, coords[1]-30), 
                                  (245, 117, 16), -1)
                    cv2.putText(image, body_language_class, coords, 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                    
                    # Get status box
                    cv2.rectangle(image, (0,0), (250, 60), (245, 117, 16), -1)
                    
                    # Display Class
                    cv2.putText(image, 'CLASS'
                                , (95,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                    cv2.putText(image, body_language_class.split(' ')[0]
                                , (90,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                    
                    # Display Probability
                    cv2.putText(image, 'PROB'
                                , (15,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                    cv2.putText(image, str(round(body_language_prob[np.argmax(body_language_prob)],2))
                                , (10,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

                if results.left_hand_landmarks:
                    pose = results.left_hand_landmarks.landmark
                    pose_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in pose]).flatten())
                    
                    row = pose_row

                    # Make Detections
                    X = pd.DataFrame([row])
                    body_language_class = model.predict(X)[0]
                    body_language_prob = model.predict_proba(X)[0]
                    # print(body_language_class, body_language_prob)
                    
                    # Grab ear coords
                    coords = tuple(np.multiply(
                                    np.array(
                                        (results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EAR].x, 
                                         results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EAR].y))
                                , [640,480]).astype(int))
                    
                    cv2.rectangle(image, 
                                  (coords[0], coords[1]+5), 
                                  (coords[0]+len(body_language_class)*20, coords[1]-30), 
                                  (245, 117, 16), -1)
                    cv2.putText(image, body_language_class, coords, 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                    
                    # Get status box
                    cv2.rectangle(image, (0,0), (250, 60), (245, 117, 16), -1)
                    
                    # Display Class
                    cv2.putText(image, 'CLASS'
                                , (95,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                    cv2.putText(image, body_language_class.split(' ')[0]
                                , (90,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                    
                    # Display Probability
                    cv2.putText(image, 'PROB'
                                , (15,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                    cv2.putText(image, str(round(body_language_prob[np.argmax(body_language_prob)],2))
                                , (10,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                
                if body_language_class is not None and body_language_class.split(' ')[0] == 'down':
                    decreaseVolume()
                if body_language_class is not None and body_language_class.split(' ')[0] == 'up':
                    increaseVolume()

                ret, buffer = cv2.imencode('.jpg', image)
                frame = buffer.tobytes()

                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result

    cap.release()

def video_feed(request):
    return StreamingHttpResponse(gen_frames(), content_type='multipart/x-mixed-replace; boundary=frame')