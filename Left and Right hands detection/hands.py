import cv2
import mediapipe as mp

#using a protobuf message to a dict
from google.protobuf.json_format import MessageToDict

mpHands = mp.solutions.hands
mpDraw = mp.solutions.drawing_utils

#defining the hand detection model
hands = mpHands.Hands(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.75,
    min_tracking_confidence=0.75,
    max_num_hands=2

)

#capturing the video from the webcam
cap = cv2.VideoCapture(0)

while True:

    #read frame by frame
    success, img = cap.read()

    #if the frame is not read successfully, break the loop
    if not success:
        print("Ignoring empty camera frame.")
        continue

    #flipping the img 
    img = cv2.flip(img, 1)

    #bgr to rgb
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    #process the rgb img
    results = hands.process(img_rgb)

    #if hand are present in the frame, draw the landmarks
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mpDraw.draw_landmarks(img, hand_landmarks, mpHands.HAND_CONNECTIONS)

        #if both hands are present
        if len(results.multi_hand_landmarks) == 2:

            #display both hands
            cv2.putText(img, "Both hands detected", 
                        (250, 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            #if any hand present
        else:
            for i in results.multi_handedness:

                #return the albel
                label = MessageToDict(i)['classification'][0]['label']

                if label == "Left":
                    cv2.putText(img, "Left hand detected", 
                                (250, 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
                if label == "Right":
                    cv2.putText(img, "Right hand detected", 
                                (250, 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow("Hand Detection", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

