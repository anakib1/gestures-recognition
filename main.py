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
    return True

def check_finger_configuration_C(hands_landmarks, idx):
    """Перевірка конфігурації пальців для літери C:
       - всі пальці зігнуті у формі літери C
       - великий палець зігнутий і прилягає до вказівного"""
    lm = hands_landmarks[idx].landmark
    threshold = 0.025 # Зменшений допуск для перевірок координат

    # 1) Перевірка, що пальці (вказівний-мізинець) зігнуті, але не повністю
    for i in range(1, 5): # Пропускаємо великий палець
        tip = lm[fingers_tips[i]]
        pip = lm[fingers_pips[i]]
        mcp = lm[fingers_mcps[i]]

        # --- DEBUG LOGS for Finger 1 --- REMOVED
        # if i == 1:
        #     print(f"  DEBUG Finger {i} (Index):")
        #     # Vertical check values
        #     vert_check1_val = tip.y > pip.y + threshold
        #     vert_check2_val = pip.y > mcp.y + threshold
        #     print(f"    Vertical Check 1: tip.y ({tip.y:.4f}) > pip.y ({pip.y:.4f}) + thr ({threshold}) = {vert_check1_val}")
        #     print(f"    Vertical Check 2: pip.y ({pip.y:.4f}) > mcp.y ({mcp.y:.4f}) + thr ({threshold}) = {vert_check2_val}")
        #     # Horizontal check values
        #     horiz_check_val = tip.x < pip.x - threshold
        #     print(f"    Horizontal Check: tip.x ({tip.x:.4f}) < pip.x ({pip.x:.4f}) - thr ({threshold}) = {horiz_check_val}")
        # --- END DEBUG LOGS ---

        # Кінчик нижче PIP (палець зігнутий вертикально)
        # Vertical check relaxed for pinky (i=4)
        if i != 4:
            if not (tip.y > pip.y + threshold):
                print(f'Finger {i} not correctly bent vertically (Tip vs PIP)')
                return False
        
        # Перевірка горизонтального вирівнювання для форми "C" - CORRECTED
        # Кінчик повинен бути правіше (більше x) ніж PIP для правої руки
        if not (tip.x > pip.x):
             print(f'Finger {i} not forming C shape horizontally (Tip vs PIP)')
             return False


    # 2) Перевірка великого пальця: зігнутий і близько до вказівного
    thumb_tip = lm[fingers_tips[0]]
    thumb_pip = lm[fingers_pips[0]]
    thumb_mcp = lm[fingers_mcps[0]]
    index_pip = lm[fingers_pips[1]]
    index_mcp = lm[fingers_mcps[1]]

    # Великий палець зігнутий горизонтально (Tip vs PIP) - More lenient check
    thumb_horizontal_threshold = threshold * 3  # Increased tolerance for thumb bend
    if not (thumb_tip.x > thumb_pip.x - thumb_horizontal_threshold):
        print('Thumb not bent horizontally enough for C shape')
        return False

    # Перевірка близькості великого пальця до вказівного - Using Euclidean distance
    dist_threshold = 0.5  # Increased threshold for distance check
    
    # Calculate distances to both PIP and MCP of index finger
    dist_pip = ((thumb_tip.x - index_pip.x) ** 2 + (thumb_tip.y - index_pip.y) ** 2) ** 0.5
    dist_mcp = ((thumb_tip.x - index_mcp.x) ** 2 + (thumb_tip.y - index_mcp.y) ** 2) ** 0.5
    
    # Use the minimum distance
    min_dist = min(dist_pip, dist_mcp)
    
    if min_dist > dist_threshold:
        print('Thumb too far from index finger')
        return False

    return True

def hand_is_symbol(hands_landmarks, idx=0):
    """Головна перевірка: долоня + конфігурація пальців."""
    # Поки що прибираємо перевірку check_palm_facing, оскільки вона може бути занадто строгою для "C"
    # return check_palm_facing(hands_landmarks, idx) and check_finger_configuration_C(hands_landmarks, idx)
    return check_finger_configuration_C(hands_landmarks, idx)

if __name__ == "__main__":
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