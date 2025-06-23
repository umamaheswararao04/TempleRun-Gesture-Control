import cv2
import mediapipe as mp
import pyautogui
import time

# Initialize webcam and mediapipe
cap = cv2.VideoCapture(0)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils
prev_action_time = 0

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            lm = hand_landmarks.landmark

            h, w, _ = img.shape
            landmarks = []
            for id, lm_point in enumerate(lm):
                x, y = int(lm_point.x * w), int(lm_point.y * h)
                landmarks.append((x, y))

            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            current_time = time.time()
            if current_time - prev_action_time > 1:

                # ✋ JUMP: Open palm — all fingers straight up
                if (
                    landmarks[8][1] < landmarks[6][1] and  # Index
                    landmarks[12][1] < landmarks[10][1] and  # Middle
                    landmarks[16][1] < landmarks[14][1] and  # Ring
                    landmarks[20][1] < landmarks[18][1]      # Pinky
                ):
                    print("JUMP - Palm Detected")
                    pyautogui.press("up")
                    prev_action_time = current_time

                # 👈 LEFT: Index pointing left
                elif landmarks[8][0] < landmarks[6][0] - 40:
                    print("LEFT")
                    pyautogui.press("left")
                    prev_action_time = current_time

                # 👉 RIGHT: Index pointing right
                elif landmarks[8][0] > landmarks[6][0] + 40:
                    print("RIGHT")
                    pyautogui.press("right")
                    prev_action_time = current_time

                # ✊ SLIDE: Fingers folded down (index and middle tips below joints)
                elif (
                    landmarks[8][1] > landmarks[6][1] and
                    landmarks[12][1] > landmarks[10][1]
                ):
                    print("SLIDE")
                    pyautogui.press("down")
                    prev_action_time = current_time

    cv2.imshow("Temple Run Hand Gesture Control", img)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()