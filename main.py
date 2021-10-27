import mediapipe as mp #For hand-recognition
import cv2 #For image-reading and manipulation

mp_drawing = mp.solutions.drawing_utils 
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

cap = cv2.VideoCapture(0) #Capturing frames from camera
with mp_hands.Hands(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
    
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # To improve performance, optionally mark the image as not writeable to pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) #Converting to RGB
    results = hands.process(image) #Detecting hands in the image using mediapipe
    image_hight, image_width, _ = image.shape

    # Draw the hand annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) 
    if results.multi_hand_landmarks: #If hands detected, annotating them
      for hand_landmarks in results.multi_hand_landmarks:
        mp_drawing.draw_landmarks(
            image,
            hand_landmarks, #The 20 points on hands
            mp_hands.HAND_CONNECTIONS) #Connecting those 20 points
        
    # Flip the image horizontally for a selfie-view display.
    cv2.imshow('Hand-Gesture-Control-System', cv2.flip(image, 1))
    #Breaking the loop key
    if cv2.waitKey(1) & 0xFF == ord('q'):
      break
cap.release()