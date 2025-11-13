import cv2
import mediapipe as mp
import pyautogui
import time
import numpy as np
from collections import deque
import threading

#Optional voice support (may need PyAudio + speech_recognition installed)
try:
    import speech_recognition as sr
    VOICE_AVAILABLE = True
except Exception:
    VOICE_AVAILABLE = False

display_w, display_h = 957, 650 #this value setted according to window size

# CONFIGURABLE PARAMETERS

SMOOTHING_COUNT = 5            # moving average window for cursor stabilization
BLINK_THRESHOLD = 0.004        # threshold for (top_y - bottom_y) to consider closed
#LONG_BLINK_TIME = 2.0          # seconds for a long blink -> right click
#BOTH_BLINK_MAX_GAP = 1.5      # seconds within which two both-eye blinks count as double-click

DWELL_TIME = 1.5              # seconds to dwell on a virtual-key for selection
MOVE_DURATION = 0.03           # pyautogui.moveTo duration (small smoothing)

CALIBRATION_POINTS = [
    ("Top-Left", (0, 0)),
    ("Top-Right", (1, 0)),
    ("Bottom-Right", (1, 1)),
    ("Bottom-Left", (0, 1)),
]
# ----------------------------

# Initialize
cam = cv2.VideoCapture(0)
# Try setting a smaller resolution for performance if needed
# cam.set(cv2.CAP_PROP_FRAME_WIDTH, 540)
# cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 960)

mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(refine_landmarks=True)

screen_w, screen_h = pyautogui.size()

# Create window (macOS may ignore resize)
# cv2.namedWindow('iGaze', cv2.WINDOW_NORMAL)
# cv2.resizeWindow('iGaze', 800, 600)

# For cursor smoothing
smooth_queue = deque(maxlen=SMOOTHING_COUNT)

# For calibration: mapping from camera normalized coords to screen coords.
# We'll compute an affine-like linear mapping using least squares:
# [sx sy]^T = A @ [fx fy 1]^T where A is 2x3. We solve for A.
calibrated_A = None
is_calibrated = False

# Blink tracking state
left_closed = False
right_closed = False
left_closed_start = None
right_closed_start = None

last_both_blink_time = 0.0  # to detect double both-eye blinks
both_blink_in_progress = False

# Virtual keyboard state
virtual_keyboard_on = True 
vk_layout = [
    list("QWERTYUIOP"),
    list("ASDFGHJKL"),
    list("ZXCVBNM"),
    ["SPACE", "BACK", "ENTER"]
]
vk_cell_w = 120
vk_cell_h = 80
vk_origin = (100, 350)
vk_dwell_start = None
vk_selected = None

# Voice command thread (optional)
def voice_thread_fn():
    r = sr.Recognizer()
    mic = sr.Microphone()
    print("[Voice] Listening for commands: 'click', 'double click', 'scroll down', 'scroll up', 'open'")
    with mic as source:
        r.adjust_for_ambient_noise(source)
    while True:
        try:
            with mic as source:
                audio = r.listen(source, phrase_time_limit=4)
            cmd = r.recognize_google(audio).lower()
            print("[Voice] Heard:", cmd)
            if "click" in cmd and "double" in cmd:
                pyautogui.doubleClick()
            elif "double click" in cmd or "double-click" in cmd:
                pyautogui.doubleClick()
            elif "click" in cmd:
                pyautogui.click()
            elif "click two times" in cmd:
                pyautogui.click()
                pyautogui.click()    
            elif "right click" in cmd:
                pyautogui.rightClick()
            elif "scroll down" in cmd:
                pyautogui.scroll(-10)
            elif "scroll up" in cmd:
                pyautogui.scroll(10)
            elif "enter" in cmd:
                # just an example: press enter
                pyautogui.press('enter')
            elif "open chrome" in cmd:
                pyautogui.open("https://www.google.com")
            # elif "something" in cmd:
            #     pyautogui.hotkey('cmd')    
        except Exception as e:
            # ignore recognition errors silently
            # print("[Voice] error:", e)
            time.sleep(0.2)

if VOICE_AVAILABLE:
    t = threading.Thread(target=voice_thread_fn, daemon=True)
    t.start()
else:
    print("[Voice] speech_recognition or microphone not available. Voice commands disabled.")

# Utility: convert normalized mediapipe landmark to pixel (frame) coords
def lm_to_pixel(landmark, frame_w, frame_h):
    return int(landmark.x * frame_w), int(landmark.y * frame_h)

# Utility: map normalized frame coords -> screen coords using calibrated_A if available
def frame_to_screen_norm(nx, ny):
    """
    nx, ny: normalized coordinates relative to frame (0..1)
    returns screen_x, screen_y
    """
    global calibrated_A, is_calibrated
    if is_calibrated and calibrated_A is not None:
        vec = np.array([nx, ny, 1.0])
        sx, sy = calibrated_A.dot(vec)
        return float(sx), float(sy)
    else:
        # naive mapping (assumes landmarks normalized in camera same orientation as screen)
        return screen_w * nx, screen_h * ny

# Calibration routine - interactive: press SPACE when looking at each corner
def calibrate_procedure():
    global calibrated_A, is_calibrated
    print("[Calibration] Starting. For each corner, look at the corner and press SPACE to record.")
    collected = []
    frame_coords = []
    screen_coords = []
    cam_local = cam  # using same capture
    for label, (sx_norm, sy_norm) in CALIBRATION_POINTS:
        print(f"[Calibration] Look at {label} corner then press SPACE (or 'q' to cancel).")
        recorded = False
        while True:
            ret, frame = cam_local.read()
            if not ret:
                continue
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            output = face_mesh.process(rgb)
            h, w, _ = frame.shape
            disp = frame.copy()
            cv2.putText(disp, f"Look at {label} - then press SPACE", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
            if output.multi_face_landmarks:
                lm = output.multi_face_landmarks[0].landmark[474]  # right iris center
                px, py = lm_to_pixel(lm, w, h)
                cv2.circle(disp, (px, py), 5, (0,255,0), -1)
            cv2.imshow('iGaze - Calibration', disp)
            k = cv2.waitKey(1) & 0xFF
            if k == ord('q'):
                print("[Calibration] Cancelled.")
                cv2.destroyWindow('iGaze - Calibration')
                return False
            if k == 32:  # SPACE
                if output.multi_face_landmarks:
                    lm = output.multi_face_landmarks[0].landmark[474]
                    # record normalized frame coords and expected screen coords
                    frame_coords.append([lm.x, lm.y, 1.0])
                    screen_coords.append([sx_norm * screen_w, sy_norm * screen_h])
                    print(f"[Calibration] Recorded {label}: frame({lm.x:.3f},{lm.y:.3f}) -> screen({sx_norm*screen_w:.1f},{sy_norm*screen_h:.1f})")
                    recorded = True
                    time.sleep(0.4)
                    break
                else:
                    print("[Calibration] No face/iris detected. Try again.")
    cv2.destroyWindow('iGaze - Calibration')

    # Solve linear least squares for a 2x3 matrix A such that A @ frame_coords^T ~= screen_coords^T
    F = np.array(frame_coords)  # Nx3
    S = np.array(screen_coords)  # Nx2
    # Solve for A.T in least squares: F @ A.T = S -> A.T = lstsq(F, S)
    A_t, *_ = np.linalg.lstsq(F, S, rcond=None)
    A = A_t.T  # 2x3
    calibrated_A = A
    is_calibrated = True
    print("[Calibration] Completed. Mapping matrix computed.")
    return True

# Helper: draw virtual keyboard
def draw_virtual_keyboard(frame):
    global vk_layout, vk_cell_w, vk_cell_h, vk_origin
    ox, oy = vk_origin
    for r_idx, row in enumerate(vk_layout):
        for c_idx, key in enumerate(row):
            x = ox + c_idx * vk_cell_w
            y = oy + r_idx * vk_cell_h
            cv2.rectangle(frame, (x, y), (x + vk_cell_w - 10, y + vk_cell_h - 10), (50, 50, 50), -1)
            cv2.rectangle(frame, (x, y), (x + vk_cell_w - 10, y + vk_cell_h - 10), (200, 200, 200), 2)
            cv2.putText(frame, key, (x + 10, y + vk_cell_h//2 + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)
    return frame

def get_vk_key_at(screen_x, screen_y):
    ox, oy = vk_origin
    for r_idx, row in enumerate(vk_layout):
        for c_idx, key in enumerate(row):
            x = ox + c_idx * vk_cell_w
            y = oy + r_idx * vk_cell_h
            if x <= screen_x <= x + vk_cell_w - 10 and y <= screen_y <= y + vk_cell_h - 10:
                return (r_idx, c_idx, key)
    return None

# Main loop
print("Controls: 'q' to quit, 'c' to calibrate, 'v' toggle virtual keyboard.")
print("Left-eye single blink -> single click. Both-eyes blink -> double click.")
print("Left-eye long blink (>0.8s) -> right click.")

while True:
    ret, frame = cam.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    output = face_mesh.process(rgb)

    if output.multi_face_landmarks:
        landmarks = output.multi_face_landmarks[0].landmark

        # Iris centers (right: 474..477, left: 469..472) -> take mean
        rx = np.mean([landmarks[i].x for i in range(474, 478)])
        ry = np.mean([landmarks[i].y for i in range(474, 478)])
        lx = np.mean([landmarks[i].x for i in range(469, 473)])
        ly = np.mean([landmarks[i].y for i in range(469, 473)])

        # Draw iris centers on frame for feedback
        rx_px, ry_px = int(rx * w), int(ry * h)
        lx_px, ly_px = int(lx * w), int(ly * h)
        cv2.circle(frame, (rx_px, ry_px), 4, (0,255,0), -1)
        cv2.circle(frame, (lx_px, ly_px), 4, (0,255,0), -1)

        # Combined gaze: average of left and right iris centers (normalized)
        gaze_nx = (rx + lx) / 2.0
        gaze_ny = (ry + ly) / 2.0

        # Convert to screen coordinates using calibration if available
        screen_x, screen_y = frame_to_screen_norm(gaze_nx, gaze_ny)

        # Stabilize with moving average
        smooth_queue.append((screen_x, screen_y))
        avg_x = sum([p[0] for p in smooth_queue]) / len(smooth_queue)
        avg_y = sum([p[1] for p in smooth_queue]) / len(smooth_queue)

        # Move cursor
        try:
            pyautogui.moveTo(avg_x, avg_y, duration=MOVE_DURATION)
        except Exception:
            # may fail on some OSes if permissions not set
            pass

        # Eyelid landmarks for blink detection
        # Left eye: upper 145, lower 159
        # Right eye: upper 374, lower 386
        l_up = landmarks[145]; l_low = landmarks[159]
        r_up = landmarks[374]; r_low = landmarks[386]
        left_dist = l_up.y - l_low.y
        right_dist = r_up.y - r_low.y

        now = time.time()

        #left eye
        if left_dist < BLINK_THRESHOLD:
            if not left_closed:
                left_closed = True 
                left_closed_start = now 

        else:
            if left_closed:
                dur = now - (left_closed_start or now)
                if dur >= 3 :
                    print("left long blink -> scroll up")
                    pyautogui.scroll(10)
                elif(dur >= 2 and dur < 3):
                    print("left short blink -> right click")
                    pyautogui.rightClick()
                #time.sleep(0.4)

            left_closed = False
            left_closed_start = None 

        if right_dist < BLINK_THRESHOLD:
            if not right_closed:
                right_closed = True 
                right_closed_start = now
        else:
            if right_closed:
                dur = now - (right_closed_start or now)
                if dur > 1.0 and dur <= 2:
                    print("right long blink -> scroll down")
                    pyautogui.scroll(-10)
                elif dur > 0.5 and dur <= 1.0:
                    print("right short blink -> left click")
                    pyautogui.click()

                #time.sleep(0.5)
            right_closed =False
            right_closed_start = None    


        # Virtual keyboard handling
        if virtual_keyboard_on:
            # Draw keyboard overlay
            kv_frame = draw_virtual_keyboard(frame)
            # Determine which key the cursor is pointing to (use avg_x, avg_y scaled to frame coords)
            # First map avg screen coords back to frame normalized coords using inverse if calibrated
            # For simplicity: use frame gaze position rx,ry to map to frame pixels and check against keyboard frame
            # We'll convert screen coords to the display coordinates for keyboard selection: approximate
            # transform: since vk is drawn on frame, we can test against gaze pixel pos (rx_px, ry_px)

            #key_info = get_vk_key_at(rx_px, ry_px)
            # Map screen coordinates (avg_x, avg_y) to frame window coordinates
            frame_x = int((avg_x / display_w) * w)
            frame_y = int((avg_y / display_h) * h)

            # Use frame-based coordinates for keyboard hover detection
            key_info = get_vk_key_at(frame_x, frame_y)



            if key_info:
                r_idx, c_idx, key = key_info
                # highlight hovered key
                x = vk_origin[0] + c_idx * vk_cell_w
                y = vk_origin[1] + r_idx * vk_cell_h
                cv2.rectangle(kv_frame, (x, y), (x + vk_cell_w - 10, y + vk_cell_h - 10), (0, 120, 255), 4)
                if vk_selected != key:
                    vk_dwell_start = time.time()
                    vk_selected = key
                else:
                    # check dwell time
                    if vk_dwell_start and (time.time() - vk_dwell_start) >= DWELL_TIME:
                        # "press" key
                        print("[VK] selected:", key)
                        if key == "SPACE":
                            pyautogui.press('space')
                        elif key == "BACK":
                            pyautogui.press('backspace')
                        elif key == "ENTER":
                            pyautogui.press('enter')    
                        else:
                            pyautogui.typewrite(key.lower())
                        vk_dwell_start = None
                        vk_selected = None
                        time.sleep(0.3)
            else:
                vk_selected = None
                vk_dwell_start = None

            frame = kv_frame

    else:
        # No face detected: small visual hint
        cv2.putText(frame, "No face detected", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

    cv2.imshow('iGaze', frame)
    k = cv2.waitKey(1) & 0xFF
    if k == ord('q'):
        break
    elif k == ord('c'):
        # calibrate routine (blocking)
        ok = calibrate_procedure()
        if ok:
            print("[Main] Calibration OK.")
    elif k == ord('v'):
        virtual_keyboard_on = not virtual_keyboard_on
        print("[Main] Virtual keyboard:", "ON" if virtual_keyboard_on else "OFF")

# cleanup
cam.release()
cv2.destroyAllWindows()

