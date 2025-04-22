import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# Індекси ключових точок
wrist_idx = [0]
# пальці
fingers_tips = [4, 8, 12, 16, 20]
fingers_dips = [3, 7, 11, 15, 19]
fingers_pips = [2, 6, 10, 14, 18]
fingers_mcps = [1, 5, 9, 13, 17]

def get_lm_coords(landmarks, indices):
    """Отримати (x, y, z) для заданих індексів."""
    return [(lm.x, lm.y, lm.z) for lm in (landmarks.landmark[idx] for idx in indices)]

def check_values(arr, rate):
    """Упевнитися, що всі сусідні значення в списку розходяться не більше ніж rate."""
    arr_sorted = sorted(arr)
    return all(abs(arr_sorted[i] - arr_sorted[i+1]) <= rate for i in range(len(arr_sorted)-1))

def check_palm_facing(hands_landmarks, idx):
    """Перевірка положення долоні у просторі."""
    # 1) Без повороту навколо Z (показник – MCP інших пальців)
    palm_mcp = get_lm_coords(hands_landmarks[idx], fingers_mcps)[1:]
    z_vals = [c[2] for c in palm_mcp]
    if not check_values(z_vals, rate=0.05):
        print('Palm rotated')
        return False
    # 2) Долоня не перевернута спинкою
    wrist_z = get_lm_coords(hands_landmarks[idx], wrist_idx)[0][2]
    thumb_z = get_lm_coords(hands_landmarks[idx], [4])[0][2]
    if wrist_z < thumb_z:
        print('Palm back')
        return False
    # 3) Без обертання по X (зап’ясток та основа вказівного пальця)
    coords = get_lm_coords(hands_landmarks[idx], wrist_idx + [fingers_tips[1]])
    if not check_values([c[2] for c in coords], rate=0.1):
        print('Tilted on X-axis')
        return False
    return True

def check_finger_configuration_G(hands_landmarks, idx):
    """Перевірка витягнутого вказівного і великого пальців під прямим кутом,
       а також зігнутих інших пальців."""
    lm = hands_landmarks[idx].landmark

    # 1) Вказівний палець має бути прямо
    y_mcp, y_pip, y_dip, y_tip = (lm[fingers_mcps[1]].y, lm[fingers_pips[1]].y,
                                 lm[fingers_dips[1]].y, lm[fingers_tips[1]].y)
    if not (y_mcp >= y_pip >= y_dip >= y_tip):
        print('Index not straight')
        return False

    # 2) Середній, безіменний, мізинець мають бути зігнуті
    for tip, pip in zip(fingers_tips[2:], fingers_pips[2:]):
        if lm[tip].y <= lm[pip].y:
            print('Some finger not bent')
            return False

    # 3) Великий палець витягнутий у бік (для правої руки – вліво від користувача)
    x_pip, x_dip, x_tip = (lm[fingers_pips[0]].x, lm[fingers_dips[0]].x, lm[fingers_tips[0]].x)
    if not (x_pip < x_dip < x_tip):
        print('Thumb not extended')
        return False

    # 4) Перевірка прямого кута між вказівним і великим пальцем
    tip_idx = get_lm_coords(hands_landmarks[idx], [fingers_tips[1]])[0]
    tip_thumb = get_lm_coords(hands_landmarks[idx], [fingers_tips[0]])[0]
    # приблизно однакова різниця по Z
    if abs(tip_idx[2] - tip_thumb[2]) > 0.1:
        print('Wrong angle')
        return False

    return True

def hand_is_symbol(hands_landmarks, idx=0):
    """Головна перевірка: долоня + конфігурація пальців."""
    return check_palm_facing(hands_landmarks, idx) and check_finger_configuration_G(hands_landmarks, idx)

# Запуск відео-стріму
cap = cv2.VideoCapture(0)
with mp_hands.Hands(
    model_complexity=0,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as hands:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.multi_hand_landmarks:
            state = hand_is_symbol(results.multi_hand_landmarks)
            color = (0, 255, 0) if state else (0, 0, 255)
            style = mp_drawing.DrawingSpec(color=color, thickness=2)
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    style
                )

        cv2.imshow('MediaPipe Hands', cv2.flip(image, 1))
        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()