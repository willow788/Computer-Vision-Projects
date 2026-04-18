import cv2
import time
import mediapipe as mp

def main():
    if not hasattr(mp, "solutions"):
        print(
            "This script requires MediaPipe Solutions API (mp.solutions), "
            "but your installed mediapipe build exposes only the Tasks API.\n"
            "Install a compatible version, for example: pip install \"mediapipe<0.10.20\""
        )
        return

    # getting the holistic model from mediapipe
    mp_holistic = mp.solutions.holistic
    holistic_model = mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    # initialising the drawing utils from mediapipe
    mp_drawing = mp.solutions.drawing_utils
    capture = cv2.VideoCapture(0)

    if not capture.isOpened():
        print("Error opening webcam.")
        holistic_model.close()
        return

    # initialising time for FPS
    previous_time = time.time()

    while capture.isOpened():
        # capture the video frame by frame
        ret, frame = capture.read()
        if not ret:
            print("Can't receive frame. Exiting ...")
            break

        # resized
        frame = cv2.resize(frame, (800, 600))

        # convert the color space from BGR to RGB
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # making prediction using the holistic model
        img.flags.writeable = False
        results = holistic_model.process(img)
        img.flags.writeable = True

        # convert the color space back to BGR
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # drawing the face landmarks on the image
        mp_drawing.draw_landmarks(
            img,
            results.face_landmarks,
            mp_holistic.FACEMESH_TESSELATION,
            mp_drawing.DrawingSpec(color=(255, 0, 255), thickness=1, circle_radius=1),
            mp_drawing.DrawingSpec(color=(255, 0, 255), thickness=1, circle_radius=1)
        )

        # drawing the right hand landmarks on the image
        mp_drawing.draw_landmarks(
            img,
            results.right_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS,
        )

        # drawing the left hand landmarks on the image
        mp_drawing.draw_landmarks(
            img,
            results.left_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS,
        )

        # calculating the FPS
        current_time = time.time()
        delta = current_time - previous_time
        fps = 0 if delta <= 0 else 1 / delta
        previous_time = current_time

        # displaying the FPS on the image
        cv2.putText(img, f'FPS: {int(fps)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # display the image
        cv2.imshow('Landmark Detection', img)

        # exiting process when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    capture.release()
    holistic_model.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
