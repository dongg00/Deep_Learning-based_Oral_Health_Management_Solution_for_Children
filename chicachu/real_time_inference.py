import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
from collections import deque
import time
from PIL import ImageFont, ImageDraw, Image

# Matplotlib 관련 라이브러리
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm # 폰트 직접 지정을 위한 라이브러리

# Matplotlib 백엔드를 'Agg'로 설정하여 GUI 창이 뜨는 것을 방지
matplotlib.use('Agg')

# m1_feature_extraction_answer.py 파일이 같은 폴더에 있어야 합니다.
import m1_feature_extraction_answer as fe

# -----------------------------------------------------------------------------
# 사용자 설정 영역 (★★★★★ 여기에 값을 꼭 입력해주세요 ★★★★★)
# -----------------------------------------------------------------------------

# 1. config.py에 정의된 MANUAL_CONFIG
cfg = {
    'feature_extraction': {
        'FEATURE_CONFIG': {
            'hand_all': True, 'face_roll_scale': True, 'face_pitch': True,
            'face_mouth_corners': True, 'face_inner_eyes': False,
        },
        'WINDOW_SIZE': 30, 'STRIDE': 3, 'USE_RAW_LANDMARK_CACHE': True,
        'INTERPOLATE_MISSING': True,
        'SMOOTHING': {
            'enabled': True, 'method': 'savgol', 'window_length': 7,
            'polyorder': 2, 'ema_alpha': 0.2,
        },
        'TEMPORAL_FEATURES': {
            'velocity': True, 'acceleration': False, 'segment_stats': True,
        },
    },
}

# 2. 가장 성능이 좋았던 모델 파일의 경로
MODEL_PATH = 'outputs/manual_run/models/fold_1/best_model.keras' # FIXME: 본인 모델 경로로 수정

# 3. 클래스 이름 (순서 중요)
CLASS_NAMES = [
    '오른쪽-구개측', '오른쪽-설측', '오른쪽-아래-씹는면', '오른쪽-위-씹는면', '오른쪽-협측',
    '왼쪽-구개측', '왼쪽-설측', '왼쪽-아래-씹는면', '왼쪽-위-씹는면', '왼쪽-협측',
    '중앙-구개측', '중앙-설측', '중앙-협측'
]

# 4. 사용할 한글 폰트 파일의 전체 경로 (Windows '맑은 고딕' 기준)
FONT_PATH = 'C:/Windows/Fonts/malgun.ttf' # FIXME: macOS 등 다른 OS의 경우 폰트 경로 수정

# 5. 추론 설정
WINDOW_SIZE = cfg['feature_extraction']['WINDOW_SIZE']
INFERENCE_STRIDE = 15
TOTAL_RUN_TIME_SECONDS = 210

# -----------------------------------------------------------------------------
# 코드 본문 (이 아래는 수정할 필요 없음)
# -----------------------------------------------------------------------------

# MediaPipe 초기화
mp_hands = mp.solutions.hands.Hands(
    max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
    max_num_faces=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# 모델 로드
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print(f"✅ 모델 로드 성공: {MODEL_PATH}")
except Exception as e:
    print(f"❌ 모델 로드 실패: {e}")
    exit()

# 필요한 얼굴 랜드마크 인덱스 미리 계산
face_indices = fe.get_face_indices_from_config(cfg)
print(f"ℹ️ 사용될 얼굴 랜드마크 인덱스: {face_indices}")

def create_radar_chart(labels, values, font_path):
    num_vars = len(labels)
    try:
        font_prop = fm.FontProperties(fname=font_path, size=8)
    except RuntimeError:
        print(f"경고: Matplotlib에서 '{font_path}' 폰트를 찾지 못했습니다. 기본 폰트를 사용합니다.")
        font_prop = fm.FontProperties(size=8)

    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]
    values_list = values.tolist()
    values_list += values_list[:1]
    fig, ax = plt.subplots(figsize=(4, 4), subplot_kw=dict(polar=True))
    max_val = max(10, np.max(values) + 5)
    ax.set_ylim(0, max_val)
    ax.plot(angles, values_list, color='red', linewidth=1)
    ax.fill(angles, values_list, color='red', alpha=0.4)
    ax.set_thetagrids(np.degrees(angles[:-1]), labels, fontproperties=font_prop)
    ax.set_yticklabels([])
    fig.canvas.draw()
    img_rgba = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
    img_rgba = img_rgba.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    plt.close(fig)
    img_rgb = cv2.cvtColor(img_rgba, cv2.COLOR_RGBA2RGB)
    return cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

def draw_text_on_frame(frame, text, position, font, color=(255, 255, 255)):
    pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_image)
    draw.text(position, text, font=font, fill=color)
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

def preprocess_buffer(buffer_data):
    raw = {'hand': np.stack([d['hand'] for d in buffer_data], axis=0), 'face_all': np.stack([d['face_all'] for d in buffer_data], axis=0) }
    raw['hand'] = fe.interpolate_nan_along_time(raw['hand']); raw['face_all'] = fe.interpolate_nan_along_time(raw['face_all'])
    sm_cfg = cfg['feature_extraction'].get('SMOOTHING', {});
    if sm_cfg.get('enabled', False):
        sm_args = {k: v for k, v in sm_cfg.items() if k != 'enabled'}; raw['hand'] = fe.smooth_timeseries(raw['hand'], **sm_args); raw['face_all'] = fe.smooth_timeseries(raw['face_all'], **sm_args)
    seq = fe.normalize_sequence(raw, cfg, face_indices); seq = fe.add_temporal_features(seq, cfg)
    if cfg['feature_extraction'].get('TEMPORAL_FEATURES', {}).get('segment_stats', False):
        mean = seq.mean(axis=0, keepdims=True); std = seq.std(axis=0, keepdims=True); repeated_mean = np.repeat(mean, seq.shape[0], axis=0); repeated_std = np.repeat(std, seq.shape[0], axis=0); seq = np.concatenate([seq, repeated_mean, repeated_std], axis=1)
    return seq

def main():
    cap = cv2.VideoCapture(0) # FIXME: 본인 카메라 번호로 수정 (0, 1, 2...)
    if not cap.isOpened():
        print("❌ 카메라를 열 수 없습니다.")
        return

    try:
        font_large = ImageFont.truetype(FONT_PATH, 24)
        font_medium = ImageFont.truetype(FONT_PATH, 20)
        font_small = ImageFont.truetype(FONT_PATH, 16)
        print(f"✅ 폰트 로드 성공: {FONT_PATH}")
    except IOError:
        print(f"❌ Pillow 폰트 파일을 찾을 수 없습니다: {FONT_PATH}")
        return

    landmark_buffer = deque(maxlen=WINDOW_SIZE)
    predictions_history = deque(maxlen=5)
    prediction_counts = np.zeros(len(CLASS_NAMES), dtype=int)
    frame_counter = 0
    start_time = time.time()
    current_action = "초기화 중..."
    action_confidence = 0.0

    while True:
        elapsed_time = time.time() - start_time
        if elapsed_time > TOTAL_RUN_TIME_SECONDS:
            print("✅ 5분 실행이 완료되어 자동으로 종료합니다.")
            break

        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False

        hand_results = mp_hands.process(frame_rgb)
        face_results = mp_face_mesh.process(frame_rgb)

        current_landmarks = {
            'hand': np.full((fe.NUM_HAND_LMS, 2), np.nan, dtype=np.float32),
            'face_all': np.full((fe.NUM_FACE_LMS, 2), np.nan, dtype=np.float32)
        }

        if hand_results.multi_hand_landmarks:
            hand_lm = hand_results.multi_hand_landmarks[0]
            current_landmarks['hand'] = np.array([[lm.x, lm.y] for lm in hand_lm.landmark], dtype=np.float32)
            mp.solutions.drawing_utils.draw_landmarks(frame, hand_lm, mp.solutions.hands.HAND_CONNECTIONS)

        if face_results.multi_face_landmarks:
            face_lm = face_results.multi_face_landmarks[0]
            current_landmarks['face_all'] = np.array([[lm.x, lm.y] for lm in face_lm.landmark], dtype=np.float32)

        landmark_buffer.append(current_landmarks)
        frame_counter += 1

        if len(landmark_buffer) == WINDOW_SIZE and frame_counter >= INFERENCE_STRIDE:
            frame_counter = 0
            processed_window = preprocess_buffer(list(landmark_buffer))
            input_tensor = np.expand_dims(processed_window, axis=0)
            prediction = model.predict(input_tensor, verbose=0)[0]
            pred_index = np.argmax(prediction)
            action_confidence = prediction[pred_index]
            
            if action_confidence > 0.5:
                prediction_counts[pred_index] += 1
                current_action = CLASS_NAMES[pred_index]
            else:
                current_action = "불확실"

            timestamp = time.strftime("%H:%M:%S")
            log_entry = f"[{timestamp}] {current_action} ({action_confidence:.2f})"
            predictions_history.append(log_entry)

        panel = np.zeros((frame.shape[0], 400, 3), dtype=np.uint8)
        radar_chart_img = create_radar_chart(CLASS_NAMES, prediction_counts, FONT_PATH)
        radar_chart_resized = cv2.resize(radar_chart_img, (380, 380))
        panel[100:480, 10:390] = radar_chart_resized

        remaining_time = TOTAL_RUN_TIME_SECONDS - elapsed_time
        mins, secs = divmod(int(remaining_time), 60)
        timer_text = f"남은 시간: {mins:02d}:{secs:02d}"
        panel = draw_text_on_frame(panel, timer_text, (10, 10), font_medium, (0, 165, 255))
        panel = draw_text_on_frame(panel, "현재 예측", (10, 40), font_medium, (255, 255, 255))
        
        color_bgr = (0, 255, 0) if action_confidence > 0.7 and current_action != "불확실" else (0, 255, 255)
        color_rgb = (color_bgr[2], color_bgr[1], color_bgr[0])
        panel = draw_text_on_frame(panel, current_action, (10, 65), font_large, color_rgb)
        
        panel = draw_text_on_frame(panel, "예측 기록", (10, 500), font_medium, (255, 255, 255))
        for i, log in enumerate(reversed(predictions_history)):
             y_pos = 530 + i * 25
             if y_pos < panel.shape[0]:
                panel = draw_text_on_frame(panel, log, (10, y_pos), font_small, (200, 200, 200))

        combined_frame = np.concatenate((frame, panel), axis=1)
        cv2.imshow('실시간 양치질 부위 인식', combined_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("사용자가 'q'를 눌러 프로그램을 종료합니다.")
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()