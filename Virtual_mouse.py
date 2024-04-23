import cv2
import mediapipe as mp
import mouse
import math

def map_number(num, in_min, in_max, out_min, out_max):
    # Map the number from the input range to the output range using linear interpolation
    return (num - in_min) * (out_max - out_min) / (in_max - in_min) + out_min

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

# Initialize OpenCV
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)
cap.set(6, cv2.VideoWriter.fourcc('M', 'J', 'P', 'G'))

while cap.isOpened():
    success, image = cap.read()
    if not success:
        continue
    img = cv2.flip(image,1)
    # Convert the image to RGB
    image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Process the image to detect hands
    results = hands.process(image_rgb)

    if results.multi_hand_landmarks:
        # If hands are detected, loop through each detected hand
        for hand_landmarks in results.multi_hand_landmarks:
            # Get the coordinates of the tip of the index finger (landmark 8)
            index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            index_finger_x = int(index_finger_tip.x * image.shape[1])
            index_finger_y = int(index_finger_tip.y * image.shape[0])

            # Get the coordinates of the tip of the big finger (landmark 4)
            big_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            big_finger_x = int(big_finger_tip.x * image.shape[1])
            big_finger_y = int(big_finger_tip.y * image.shape[0])

            # Calculate the distance between the two finger tips
            distance = math.sqrt((index_finger_x - big_finger_x) ** 2 + (index_finger_y - big_finger_y) ** 2)

            # Define a threshold for touch detection
            touch_threshold = 30  # Adjust this value as needed

            # If the distance is less than the threshold, perform a click action
            if distance < touch_threshold:
                mouse.click()

            # Map the index finger coordinates to the screen size
            x = map_number(index_finger_x, 0,1280, 0, 2020)
            y = map_number(index_finger_y, 0, 720, 0, 1180)

            # Move the mouse pointer to the mapped coordinates
            mouse.move(x, y)

    # Display the image
    cv2.imshow("Hand Detection", image)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the VideoCapture and destroy OpenCV windows
cap.release()
cv2.destroyAllWindows()
