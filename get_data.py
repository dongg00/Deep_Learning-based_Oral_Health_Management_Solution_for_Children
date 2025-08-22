import cv2
import mediapipe as mp
import numpy as np
import time
import os
import json
from PIL import ImageFont, ImageDraw, Image

# ===============================
# --- 1. 기본 설정 ---
# ===============================
VIDEO_DATA_PATH = os.path.join('video_data')

RECORDING_FPS = 15.0

actions = [
    '왼쪽-협측', '중앙-협측', '오른쪽-협측', '오른쪽-구개측', '중앙-구개측', '왼쪽-구개측',
    '왼쪽-설측', '중앙-설측', '오른쪽-설측', '오른쪽-위-씹는면', '왼쪽-위-씹는면',
    '왼쪽-아래-씹는면', '오른쪽-아래-씹는면'
]
record_keys = ['q', 'w', 'e', 'a', 's', 'd', 'f', 'g', 'z', 'x', 'c', 'v', 'b', 'n']
guide_keys =  ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0', '-', '=', ']']

RECORD_ACTION_MAP = {record_keys[i]: action for i, action in enumerate(actions)}
GUIDE_ACTION_MAP  = {guide_keys[i]: actions[i] for i in range(len(actions))}

for action in actions:
    os.makedirs(os.path.join(VIDEO_DATA_PATH, action), exist_ok=True)

# 폰트
try:
    font_path = "C:/Windows/Fonts/malgun.ttf"
    font = ImageFont.truetype(font_path, 15)
    status_font = ImageFont.truetype(font_path, 25)
except FileNotFoundError:
    font = ImageFont.load_default()
    status_font = ImageFont.load_default()
    print("⚠️ 경고: 한글 폰트를 찾을 수 없습니다.")

def draw_text(img, text, pos, font, color, center=False):
    img_pil = Image.fromarray(img)
    draw = ImageDraw.Draw(img_pil)
    if center:
        try:
            text_bbox = draw.textbbox((0, 0), text, font=font)
        except AttributeError:
            text_size = draw.textsize(text, font=font)
            text_bbox = (0, 0, text_size[0], text_size[1])
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        pos = (pos[0] - text_width // 2, pos[1] - text_height // 2)
    draw.text(pos, text, font=font, fill=color)
    return np.array(img_pil)

def count_videos_for_action(action_dir: str) -> int:
    root = os.path.join(VIDEO_DATA_PATH, action_dir)
    total = 0
    if not os.path.isdir(root):
        return 0
    for _, _, files in os.walk(root):
        total += sum(1 for f in files if f.lower().endswith('.mp4'))
    return total

# 화살표 키 코드 (Windows OpenCV)
KEY_LEFT  = 2424832
KEY_UP    = 2490368
KEY_RIGHT = 2555904
KEY_DOWN  = 2621440

# === 손목 각도 시각화 설정 ===
SHOW_WRIST_ARROW = False         # 화면에 화살표 표시 여부(기본 False)
ARROW_COLOR = (0, 255, 255)
ARROW_LEN   = 140                # 화살표 길이(px)
ANGLE_BG    = (30, 30, 30)
ANGLE_TEXT_COLOR = (255, 255, 0)

# 스무딩(EMA) & 점프 가드
class EMA:
    def __init__(self, alpha=0.35, init=None):
        self.alpha = alpha
        self.val = init
    def update(self, x):
        if x is None:
            return self.val
        if self.val is None:
            self.val = x
        else:
            self.val = self.alpha * x + (1 - self.alpha) * self.val
        return self.val

ema_theta = EMA(alpha=0.35, init=None)
last_theta_raw = None
MAX_DELTA_PER_FRAME = 30.0  # 프레임당 급점프 가드

# ===============================
# --- 3. 메인 ---
# ===============================
def main():
    GUIDE_VIDEO_PATH = 'guide_videos'
    CROSSHAIR_ARM_LENGTH = 50

    REFERENCE_DISTANCE_KEY = 'c'
    DISTANCE_TOLERANCE = 0.1
    reference_eye_distance = 91.82

    guide_video_caps = {}
    for key, action in GUIDE_ACTION_MAP.items():
        for ext in ['.mp4', '.avi', '.mov', '.wmv']:
            p = os.path.join(GUIDE_VIDEO_PATH, f"{action}{ext}")
            if os.path.exists(p):
                guide_video_caps[action] = cv2.VideoCapture(p)
                break

    label_counts = {action: count_videos_for_action(action) for action in actions}

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ 0번 카메라를 열 수 없습니다.")
        return
    print("✅ 0번 카메라에 연결되었습니다.")
    
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    if actual_fps <= 1.0:
        actual_fps = 30.0
    print(f"✅ 카메라 실제 FPS(드라이버 보고): {actual_fps:.2f}")

    mp_face_mesh = mp.solutions.face_mesh
    mp_hands     = mp.solutions.hands
    mp_drawing   = mp.solutions.drawing_utils

    with mp_face_mesh.FaceMesh(max_num_faces=2, refine_landmarks=True,
                               min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh, \
         mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.5,
                        min_tracking_confidence=0.5) as hands:

        is_recording = False
        current_label = None
        start_time = 0.0

        current_person_id = "P01"
        typing_person = False
        person_buffer = ""

        current_guide_cap = None
        selected_guide_label = "가이드 선택: 숫자 키"

        feedback_message = ""
        feedback_timer = 0.0

        current_eye_distance = 0.0
        
        RECORDING_DURATION = 10
        video_writer = None
        last_saved_video = None
        
        recorded_frames = 0
        target_frames = int(round(RECORDING_DURATION * RECORDING_FPS))
        print(f"✅ 녹화 FPS: {RECORDING_FPS:.2f}, 목표 프레임 수: {target_frames} 프레임")

        show_landmarks = True
        
        EYE_LINE_RATIO_DEFAULT = 0.38
        cross_x_ratio = 0.500
        cross_y_ratio = 0.330
        
        cross_x = None
        cross_y = None
        STEP = 5

        print("▶ 십자가 조절: ↑/↓/←/→,  I/K(가로선),  O(초기화)")
        print(f"▶ 거리 캘리브레이션: '{REFERENCE_DISTANCE_KEY}' 키")
        print("▶ 저장/불러오기: S / L  (config: crosshair_config.json)")
        print("▶ 랜드마크: T 토글,  사람ID: Shift+P → 숫자 후 Enter")
        print("▶ 종료: '`' 또는 ESC")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            cam_h, cam_w, _ = frame.shape

            if cross_x is None or cross_y is None:
                cross_x = int(cross_x_ratio * cam_w)
                cross_y = int(cross_y_ratio * cam_h)

            # --- 녹화 시작 준비 ---
            if is_recording and video_writer is None:
                timestamp = int(time.time())
                person_dir = os.path.join(VIDEO_DATA_PATH, current_label, current_person_id)
                os.makedirs(person_dir, exist_ok=True)
                video_path = os.path.join(person_dir, f"{timestamp}.mp4")
                fourcc = cv2.VideoWriter_fourcc(*('m','p','4','v'))
                video_writer = cv2.VideoWriter(video_path, fourcc, RECORDING_FPS, (cam_w, cam_h))

                recorded_frames = 0
                start_time = time.time()
                last_written_frame = frame.copy()
                target_frames = int(round(RECORDING_DURATION * RECORDING_FPS))
                print(f"📹 비디오 녹화 시작: {video_path}")

            # --- 프레임 기록 (시간 스케줄링 + 누락 슬롯 보충) ---
            if is_recording and video_writer is not None:
                now = time.time()
                last_written_frame = frame  # 항상 최신 프레임 보관

                ideal_count = int((now - start_time) * RECORDING_FPS)
                while recorded_frames < ideal_count and recorded_frames < target_frames:
                    video_writer.write(last_written_frame)
                    recorded_frames += 1

            # --- 종료 조건: 시간 기준 ---
            if is_recording and (time.time() - start_time) >= RECORDING_DURATION:
                if video_writer is not None:
                    if last_written_frame is None:
                        last_written_frame = frame
                    while recorded_frames < target_frames:
                        video_writer.write(last_written_frame)
                        recorded_frames += 1

                    video_writer.release()
                    person_dir = os.path.join(VIDEO_DATA_PATH, current_label, current_person_id)
                    try:
                        files = [os.path.join(person_dir, f) for f in os.listdir(person_dir) if f.lower().endswith('.mp4')]
                        last_saved_video = max(files, key=os.path.getmtime) if files else None
                    except Exception:
                        last_saved_video = None
                    video_writer = None
                    print("📹 비디오 저장 완료")

                if current_label:
                    label_counts[current_label] = count_videos_for_action(current_label)
                is_recording = False
                feedback_message = f"저장 완료: {current_label} / {current_person_id}"
                feedback_timer = time.time()

            # --- 추론 ---
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_results = face_mesh.process(rgb)
            hand_results = hands.process(rgb)

            # --- 레이아웃 ---
            info_w, guide_area_w = 500, int(cam_w * 0.8)
            canvas = np.zeros((cam_h, cam_w + guide_area_w + info_w, 3), dtype=np.uint8)
            canvas[:cam_h, :cam_w] = frame.copy()

            cam_view = canvas[:cam_h, :cam_w]

            # --- 십자가 ---
            cross_x = int(np.clip(cross_x, 0, cam_w - 1))
            cross_y = int(np.clip(cross_y, 0, cam_h - 1))

            cv2.line(cam_view, (cross_x, cross_y - CROSSHAIR_ARM_LENGTH), (cross_x, cross_y + CROSSHAIR_ARM_LENGTH), (0, 255, 255), 2, cv2.LINE_AA)
            cv2.line(cam_view, (cross_x - CROSSHAIR_ARM_LENGTH, cross_y), (cross_x + CROSSHAIR_ARM_LENGTH, cross_y), (0, 255, 255), 2, cv2.LINE_AA)
            cv2.circle(cam_view, (cross_x, cross_y), 6, (0, 255, 0), -1, cv2.LINE_AA)

            # --- 랜드마크 ---
            if show_landmarks:
                if face_results.multi_face_landmarks:
                    for fl in face_results.multi_face_landmarks:
                        mp_drawing.draw_landmarks(
                            image=cam_view, landmark_list=fl, connections=mp_face_mesh.FACEMESH_TESSELATION,
                            landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0,255,0), thickness=1, circle_radius=1),
                            connection_drawing_spec=mp_drawing.DrawingSpec(color=(0,180,0), thickness=1)
                        )
                if hand_results.multi_hand_landmarks:
                    for hl in hand_results.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(
                            image=cam_view, landmark_list=hl, connections=mp_hands.HAND_CONNECTIONS,
                            landmark_drawing_spec=mp_drawing.DrawingSpec(color=(255,255,255), thickness=2, circle_radius=2),
                            connection_drawing_spec=mp_drawing.DrawingSpec(color=(255,255,255), thickness=2)
                        )

            # --- 상태 텍스트 ---
            face_status = "얼굴 감지됨" if face_results.multi_face_landmarks else "얼굴 감지 안됨"
            hand_status = "손 감지됨" if hand_results.multi_hand_landmarks else "손 감지 안됨"
            face_color = (0, 255, 0) if face_results.multi_face_landmarks else (0, 0, 255)
            hand_color = (0, 255, 0) if hand_results.multi_hand_landmarks else (0, 0, 255)
            canvas = draw_text(canvas, face_status, (cam_w // 2, cam_h - 60), status_font, face_color, center=True)
            canvas = draw_text(canvas, hand_status, (cam_w // 2, cam_h - 30), status_font, hand_color, center=True)

            # --- 거리 피드백 ---
            distance_feedback_text = ""
            distance_feedback_color = (255, 255, 255)

            if face_results.multi_face_landmarks:
                face_landmarks = face_results.multi_face_landmarks[0].landmark
                left_eye_x  = face_landmarks[33].x * cam_w
                left_eye_y  = face_landmarks[33].y * cam_h
                right_eye_x = face_landmarks[263].x * cam_w
                right_eye_y = face_landmarks[263].y * cam_h
                current_eye_distance = np.linalg.norm(
                    np.array([left_eye_x, left_eye_y]) - np.array([right_eye_x, right_eye_y])
                )

                if reference_eye_distance is not None:
                    lower_bound = reference_eye_distance * (1 - DISTANCE_TOLERANCE)
                    upper_bound = reference_eye_distance * (1 + DISTANCE_TOLERANCE)

                    if current_eye_distance < lower_bound:
                        distance_feedback_text = "너무 멉니다. 가까이 오세요!"
                        distance_feedback_color = (0, 0, 255)
                    elif current_eye_distance > upper_bound:
                        distance_feedback_text = "너무 가깝습니다. 뒤로 가세요!"
                        distance_feedback_color = (0, 0, 255)
                    else:
                        distance_feedback_text = "적정 거리입니다."
                        distance_feedback_color = (0, 255, 0)

            if distance_feedback_text:
                canvas = draw_text(canvas, distance_feedback_text, (cam_w // 2, 90), status_font, distance_feedback_color, center=True)

            # === 손목 굴곡각: 수평선 대비 (수직=90°, 왼쪽<90°, 오른쪽>90°), 수평선은 화면에 표시하지 않음 ===
            angle_displayed = None
            if hand_results.multi_hand_landmarks:
                hl = hand_results.multi_hand_landmarks[0]
                lms = hl.landmark

                WRIST_ID = 0
                PINKY_MCP_ID = 17  # 새끼손가락 뿌리

                wx, wy = int(lms[WRIST_ID].x * cam_w), int(lms[WRIST_ID].y * cam_h)
                px, py = int(lms[PINKY_MCP_ID].x * cam_w), int(lms[PINKY_MCP_ID].y * cam_h)

                # 손등 방향 단위벡터 (WRIST -> PINKY_MCP)
                v = np.array([px - wx, py - wy], dtype=float)
                v_norm = np.linalg.norm(v) + 1e-8
                v /= v_norm

                # 수직(위) 기준의 부호 있는 편차: 오른쪽(+) / 왼쪽(-)
                # u = (0, -1). delta = atan2(cross(u, v), dot(u, v)) = atan2(vx, -vy)
                delta = np.degrees(np.arctan2(v[0], -v[1]))   # 범위 (-180, 180]

                # 최종 각도: 수직 90°, 왼쪽<90°, 오른쪽>90°
                theta_raw = 90.0 + delta

                # 안전하게 [0, 180]로 접어넣기 (아래쪽 향하는 예외 자세 등 처리)
                theta_raw = (theta_raw + 360.0) % 360.0
                if theta_raw > 180.0:
                    theta_raw = 360.0 - theta_raw

                # 점프 가드 + EMA 스무딩
                global last_theta_raw, ema_theta
                if last_theta_raw is not None and abs(theta_raw - last_theta_raw) > MAX_DELTA_PER_FRAME:
                    theta_use = last_theta_raw
                else:
                    theta_use = theta_raw
                last_theta_raw = theta_use

                theta_s = ema_theta.update(theta_use)
                angle_displayed = None if theta_s is None else float(theta_s)

                # (옵션) 손등 화살표 표시: 기본 False
                if SHOW_WRIST_ARROW:
                    end_x = int(wx + v[0] * ARROW_LEN)
                    end_y = int(wy + v[1] * ARROW_LEN)
                    cv2.arrowedLine(cam_view, (wx, wy), (end_x, end_y), ARROW_COLOR, 5, tipLength=0.3)

                # 각도 텍스트(손목 근처 버블)
                if angle_displayed is not None:
                    txt = f"{angle_displayed:4.1f}°"
                    (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                    pad = 6
                    bx1, by1 = wx + 10, wy - th - 14
                    bx2, by2 = bx1 + tw + 2 * pad, by1 + th + 2 * pad
                    cv2.rectangle(cam_view, (bx1, by1), (bx2, by2), ANGLE_BG, -1, cv2.LINE_AA)
                    cv2.putText(cam_view, txt, (bx1 + pad, by2 - pad - 2),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, ANGLE_TEXT_COLOR, 2, cv2.LINE_AA)

            # --- 가이드 영상 영역 ---
            max_guide_w, max_guide_h = guide_area_w, cam_h - 150
            cv2.rectangle(canvas, (cam_w + 10, 40), (cam_w + 10 + max_guide_w, 40 + max_guide_h), (20, 20, 20), -1)
            if current_guide_cap:
                ret_g, g_frame = current_guide_cap.read()
                if not ret_g:
                    current_guide_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret_g, g_frame = current_guide_cap.read()
                if ret_g:
                    gh, gw, _ = g_frame.shape
                    ratio = min(max_guide_w / gw, max_guide_h / gh) if gw and gh else 0
                    tw, th = int(gw * ratio), int(gh * ratio)
                    resized = cv2.resize(g_frame, (tw, th))
                    sx = cam_w + 10 + (max_guide_w - tw) // 2
                    sy = 40 + (max_guide_h - th) // 2
                    canvas[sy:sy+th, sx:sx+tw] = resized
            else:
                canvas = draw_text(canvas, "영상을 선택하세요",
                                   (cam_w + 10 + max_guide_w // 2, 40 + max_guide_h // 2),
                                   status_font, (100, 100, 100), center=True)

            # --- 정보 패널 ---
            info_panel_x = cam_w + guide_area_w + 20
            line_h = 28
            y = 20

            canvas = draw_text(canvas, "[녹화]/[가이드]: 라벨 [파일 수]", (info_panel_x, y), font, (255,255,255)); y += 40
            for i, action in enumerate(actions):
                rec_key = record_keys[i]
                guide_key = guide_keys[i] if i < len(guide_keys) else ' '
                text = f" '{rec_key}' / '{guide_key}' : {action} [{label_counts.get(action, 0)}]"
                color = (0,255,0) if is_recording and current_label == action else (255,255,255)
                canvas = draw_text(canvas, text, (info_panel_x, y), font, color); y += line_h

            y += 10
            canvas = draw_text(canvas, f"사람 ID: {current_person_id} (Shift+P 변경)", (info_panel_x, y), font, (0,200,255)); y += line_h
            canvas = draw_text(canvas, f"T: 랜드마크 토글", (info_panel_x, y), font, (0,165,255)); y += line_h

            # --- 녹화 진행 텍스트/디버그 or 상단 HUD ---
            if typing_person:
                text = f"ID 입력 중: {person_buffer}_"
                canvas = draw_text(canvas, text, (cam_w // 2, 30), status_font, (255, 255, 0), center=True)
            elif feedback_message and (time.time() - feedback_timer) < 3.0:
                canvas = draw_text(canvas, feedback_message, (cam_w // 2, 30), status_font, (0, 255, 255), center=True)
            elif is_recording:
                elapsed_time = time.time() - start_time
                text = f"녹화 중: {current_label} ({elapsed_time:.1f}s / {RECORDING_DURATION}s)"
                canvas = draw_text(canvas, text, (cam_w // 2, 30), status_font, (0, 0, 255), center=True)
                dbg = f"Written: {recorded_frames}/{target_frames} frames  |  Target FPS: {RECORDING_FPS:g}"
                canvas = draw_text(canvas, dbg, (20, 20), font, (0, 255, 255))
            else:
                if angle_displayed is not None:
                    canvas = draw_text(canvas, f"손목 굴곡 각도(수평선 대비): {angle_displayed:4.1f}°",
                                       (cam_w // 2, 30), status_font, (0, 255, 255), center=True)
                else:
                    canvas = draw_text(canvas, selected_guide_label, (cam_w // 2, 30), status_font, (0, 255, 255), center=True)

            cv2.imshow("collect datas", canvas)

            # ---- 키 처리 ----
            key = cv2.waitKey(1)
            if key == -1:
                continue

            if key == 27 or (key & 0xFF) == ord('`'):
                break

            if key in (KEY_UP, ord('i'), ord('I')):
                cross_y -= STEP; continue
            if key in (KEY_DOWN, ord('k'), ord('K')):
                cross_y += STEP; continue
            if key == KEY_LEFT:
                cross_x -= STEP; continue
            if key == KEY_RIGHT:
                cross_x += STEP; continue
            if (key & 0xFF) in (ord('o'), ord('O')):
                cross_x = cam_w // 2
                cross_y = int(cam_h * EYE_LINE_RATIO_DEFAULT)
                continue

            if (key & 0xFF) in (ord('t'), ord('T')):
                show_landmarks = not show_landmarks
                feedback_message = f"랜드마크 표시: {'ON' if show_landmarks else 'OFF'}"
                feedback_timer = time.time()
                continue
                
            if typing_person:
                if key == 13:
                    if len(person_buffer) > 0:
                        try:
                            num = int(person_buffer)
                            current_person_id = f"P{num:02d}"
                            feedback_message = f"사람 ID 설정: {current_person_id}"
                        except ValueError:
                            feedback_message = "숫자만 입력하세요"
                    else:
                        feedback_message = "입력 취소"
                    feedback_timer = time.time()
                    typing_person = False
                    person_buffer = ""
                    continue
                elif key == 27:
                    typing_person = False; person_buffer = ""
                    feedback_message = "입력 취소"; feedback_timer = time.time()
                    continue
                elif key == 8:
                    person_buffer = person_buffer[:-1]; continue
                else:
                    ch = chr(key & 0xFF)
                    if ch.isdigit():
                        person_buffer += ch
                        if len(person_buffer) > 2:
                            person_buffer = person_buffer[:2]
                    continue

            key_char = chr(key & 0xFF)
            if is_recording:
                continue

            if key == 8:
                if last_saved_video and os.path.exists(last_saved_video):
                    try:
                        os.remove(last_saved_video)
                        parts = os.path.normpath(last_saved_video).split(os.sep)
                        if len(parts) >= 3:
                            deleted_label = parts[-3]
                            if deleted_label in label_counts and label_counts[deleted_label] > 0:
                                label_counts[deleted_label] -= 1
                        feedback_message = f"삭제: {os.path.basename(last_saved_video)}"
                    except Exception as e:
                        feedback_message = f"삭제 실패: {e}"
                else:
                    feedback_message = "삭제할 파일 없음"
                feedback_timer = time.time()
                continue

            if key_char == 'P':
                typing_person = True
                person_buffer = ""
                continue

            if key_char in RECORD_ACTION_MAP:
                current_label = RECORD_ACTION_MAP[key_char]
                is_recording = True
                start_time = time.time()
                last_saved_video = None
                print(f"▶️ 데이터 녹화 시작: {current_label} / {current_person_id}")
                continue

            if key_char in GUIDE_ACTION_MAP:
                selected_action = GUIDE_ACTION_MAP[key_char]
                current_guide_cap = guide_video_caps.get(selected_action)
                selected_guide_label = selected_action if current_guide_cap else f"'{selected_action}' 영상 없음"
                if current_guide_cap:
                    current_guide_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

        print("프로그램을 종료합니다.")
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
