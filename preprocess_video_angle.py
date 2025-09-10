# -*- coding: utf-8 -*-
"""
영상 → 프레임 피처 추출 (MediaPipe FaceMesh + Hands)
- 버전: nose 원점 + eye-기준 회전/스케일 보정 좌표계
- 프레임 피처:
  손 21 랜드마크(x,y)=42 + 얼굴 보조(인중, 양쪽 눈꼬리 x,y)=6 = 기본 48차원
  + 중지(Middle) MCP→손목 전역각 6D(절대각 2 + 수평/수직 상대각 4) + 길이 1
  + 새끼(Pinky) MCP→PIP 전역각 6D(절대각 2 + 수평/수직 상대각 4) + 길이 1
  = 총 62차원
- FPS 리샘플링(15fps), 시간적 평균화(스텝 평균)
- 입력 해상도 640×480 기준(레터박스)
- 얼굴 미검출 프레임은 스킵(데이터 품질), 손 미검출 시 손 좌표는 zero-fill

출력: <project_tag>/data_<project_tag>/<라벨>/<사람>/<클립>.npy (T×62)
"""

# ----- 필수 라이브러리 임포트 -----
import os
import cv2
import glob
import json
import math
import time
import argparse
import numpy as np
from typing import Tuple, Optional

# 시스템 인코딩 설정 (Windows 환경에서의 한글 경로 문제 방지)
os.environ["PYTHONUTF8"] = "1"

# MediaPipe 라이브러리 임포트 및 예외 처리
try:
    import mediapipe as mp
except Exception:
    raise RuntimeError("mediapipe가 설치되어 있어야 합니다: pip install mediapipe")

# -------------------- 고정 임계값 (Constants) --------------------
TARGET_FPS = 15
TARGET_W, TARGET_H = 640, 480
# 각도 계산 시 벡터 최소 길이(눈거리로 정규화된 좌표 기준)
MIN_VEC_LEN = 0.01

# -------------------- 데이터셋 관련 설정 --------------------
KOREAN_LABELS = [
    '왼쪽-협측','중앙-협측','오른쪽-협측',
    '왼쪽-구개측','중앙-구개측','오른쪽-구개측',
    '왼쪽-설측','중앙-설측','오른쪽-설측',
    '오른쪽-위-씹는면','왼쪽-위-씹는면','왼쪽-아래-씹는면','오른쪽-아래-씹는면'
]

# -------------------- FaceMesh 랜드마크 인덱스 --------------------
NOSE_TIP_IDX    = 1
LEFT_EYE_IDX    = 133
RIGHT_EYE_IDX   = 362
PHILTRUM_IDX    = 13
EYE_OUTER_L     = 33
EYE_OUTER_R     = 263
CHIN_IDX        = 152

# -------------------- 유틸리티 함수 (Helper Functions) --------------------
def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def list_videos(video_root: str):
    for label in KOREAN_LABELS:
        ldir = os.path.join(video_root, label)
        if not os.path.isdir(ldir): continue
        for person in sorted(os.listdir(ldir)):
            pdir = os.path.join(ldir, person)
            if not os.path.isdir(pdir): continue
            for ext in ("*.mp4","*.mov","*.avi","*.mkv","*.webm"):
                for v in glob.glob(os.path.join(pdir, ext)):
                    if os.path.isdir(v): continue
                    yield label, person, v

def resample_step(in_fps: float, out_fps: float) -> int:
    if in_fps is None or in_fps <= 1: return 1
    if out_fps is None or out_fps <= 0: return 1
    return max(1, int(round(in_fps / out_fps)))

def resolve_video_root(arg_root: Optional[str]) -> str:
    candidates = []
    if arg_root: candidates.append(arg_root)
    candidates.append(os.path.join(os.getcwd(), "video_data"))
    try:
        here = os.path.dirname(__file__)
    except NameError:
        here = os.getcwd()
    candidates.append(os.path.join(here, "video_data"))
    candidates.append(r"D:\chicachu\video_data") # 개인 환경에 맞게 수정
    env_root = os.environ.get("VIDEO_ROOT")
    if env_root: candidates.append(env_root)
    for c in candidates:
        if c and os.path.isdir(c):
            return c
    raise FileNotFoundError("video_data 폴더를 찾을 수 없습니다.")

# 2D 회전 행렬
def rot2d(theta: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s], [s, c]], dtype=np.float32)

# 각도 유틸
def wrap_angle(phi: float) -> float:
    return (phi + np.pi) % (2*np.pi) - np.pi

def mean_angle_toroidal(thetas: np.ndarray) -> float:
    if thetas.size == 0:
        return 0.0
    return float(np.arctan2(np.mean(np.sin(thetas)), np.mean(np.cos(thetas))))

# -------------------- 핵심 기능: 피처 추출 함수 --------------------
def extract_features_from_video(path: str, use_eye_feats: bool = True) -> np.ndarray:
    """
    단일 비디오 파일로부터 시계열 특징(feature)을 추출합니다.
    - 얼굴 정규화 좌표계(코 원점, 눈-눈 회전 정렬, 눈거리 스케일)로 변환
    - 손 21 랜드마크(x,y)=42 + 얼굴 보조(인중/눈꼬리 x,y)=6 → 기본 48차원
    - 추가: 중지 MCP→손목, 새끼 MCP→PIP 각도 피처(각 6D) + 벡터 길이(각 1D)
    - 얼굴 미검출 프레임은 스킵, 손 미검출은 zero-fill
    - 시간 평균(step 단위)으로 리샘플링
    """
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")

    in_fps = cap.get(cv2.CAP_PROP_FPS) or TARGET_FPS
    step = resample_step(in_fps, TARGET_FPS)

    face_model = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False, max_num_faces=1, refine_landmarks=True,
        min_detection_confidence=0.5, min_tracking_confidence=0.5
    )
    hand_model = mp.solutions.hands.Hands(
        static_image_mode=False, max_num_hands=1,
        min_detection_confidence=0.4, min_tracking_confidence=0.4
    )

    T = []
    feature_buffer = []

    # 각도 누적용 버퍼(프레임 → 스텝 평균)
    theta_mid_list = []     # 중지 MCP(9) → 손목(0)
    theta_pinky_list = []   # 새끼 MCP(17) → PIP(18)
    theta_horz_list = []    # 얼굴 수평축(눈-눈)
    theta_vert_list = []    # 얼굴 수직축(코-턱)
    len_mid_list   = []     # 중지 벡터 길이
    len_pinky_list = []     # 새끼 벡터 길이

    FEAT_DIM = 62  # 48 기본 + (중지 6 + 새끼 6) + 길이 2

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w, _ = frame.shape
        if h == TARGET_H and w == TARGET_W:
            padded_frame = frame
        else:
            scale_img = min(TARGET_W / w, TARGET_H / h)
            new_w, new_h = int(w * scale_img), int(h * scale_img)
            resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            padded_frame = np.full((TARGET_H, TARGET_W, 3), 0, dtype=np.uint8)
            top = (TARGET_H - new_h) // 2
            left = (TARGET_W - new_w) // 2
            padded_frame[top:top + new_h, left:left + new_w] = resized

        rgb = cv2.cvtColor(padded_frame, cv2.COLOR_BGR2RGB)
        f_res = face_model.process(rgb)
        if not f_res.multi_face_landmarks:
            continue
        h_res = hand_model.process(rgb)

        face_lm = f_res.multi_face_landmarks[0].landmark
        le = np.array([face_lm[LEFT_EYE_IDX].x,  face_lm[LEFT_EYE_IDX].y], dtype=np.float32)
        re = np.array([face_lm[RIGHT_EYE_IDX].x, face_lm[RIGHT_EYE_IDX].y], dtype=np.float32)
        nose = np.array([face_lm[NOSE_TIP_IDX].x, face_lm[NOSE_TIP_IDX].y], dtype=np.float32)

        # --- 코-턱 벡터를 이용한 피치 보정 ---
        chin = np.array([face_lm[CHIN_IDX].x, face_lm[CHIN_IDX].y], dtype=np.float32)
        pitch_vec = chin - nose
        pitch_angle = math.atan2(float(pitch_vec[1]), float(pitch_vec[0])) - (math.pi / 2)
        conf = np.linalg.norm(pitch_vec)
        c = float(np.clip((conf - 0.02) / 0.05, 0.0, 1.0))
        R_pitch = rot2d(-c * pitch_angle)
        # 눈-눈 롤 보정
        eye_vec = re - le
        roll_angle = math.atan2(float(eye_vec[1]), float(eye_vec[0]))
        R_roll = rot2d(-roll_angle)
        # 최종 회전 행렬
        R_final = R_pitch @ R_roll
        scale = float(np.linalg.norm(eye_vec) + 1e-6)

        def nrm_face(pt_idx: int) -> np.ndarray:
            p = np.array([face_lm[pt_idx].x, face_lm[pt_idx].y], dtype=np.float32) - nose
            return (R_final @ p) / scale

        phil = nrm_face(PHILTRUM_IDX)
        eL   = nrm_face(EYE_OUTER_L)
        eR   = nrm_face(EYE_OUTER_R)
        face_aux = np.array([phil[0], phil[1], eL[0], eL[1], eR[0], eR[1]], dtype=np.float32)

        hand_flat = np.zeros(42, dtype=np.float32)
        v_mid = None
        v_pinky = None

        if h_res.multi_hand_landmarks:
            hand = h_res.multi_hand_landmarks[0].landmark
            pts = []
            pts_xy = []
            for i in range(21):
                p = np.array([hand[i].x, hand[i].y], dtype=np.float32) - nose
                p = (R_final @ p) / scale
                pts.extend([p[0], p[1]])
                pts_xy.append(p)
            hand_flat = np.array(pts, dtype=np.float32)
            pts_xy = np.asarray(pts_xy, dtype=np.float32)

            # --- 각도용 벡터 계산 ---
            # 중지 MCP(9) → 손목(0): 손 전체 축 근사
            v_mid = pts_xy[0] - pts_xy[9]   # (손목 - 중지MCP) == "중지→손목" 방향
            # 새끼 MCP(17) → PIP(18): 손가락 기울기
            v_pinky = pts_xy[18] - pts_xy[17]

            # 얼굴 축 각도(프레임별)도 함께 저장
            th_h = float(np.arctan2(eye_vec[1],   eye_vec[0]))
            th_v = float(np.arctan2(pitch_vec[1], pitch_vec[0]))
        else:
            # 손 미검출 프레임: 각도는 누적 안 함
            th_h = float(np.arctan2(eye_vec[1],   eye_vec[0]))
            th_v = float(np.arctan2(pitch_vec[1], pitch_vec[0]))

        # 길이 계산 및 유효 프레임만 각도 누적
        if v_mid is not None:
            len_mid = float(np.linalg.norm(v_mid) + 1e-12)
            if len_mid >= MIN_VEC_LEN:
                theta_mid_list.append(float(np.arctan2(v_mid[1], v_mid[0])))
                len_mid_list.append(len_mid)
        if v_pinky is not None:
            len_pinky = float(np.linalg.norm(v_pinky) + 1e-12)
            if len_pinky >= MIN_VEC_LEN:
                theta_pinky_list.append(float(np.arctan2(v_pinky[1], v_pinky[0])))
                len_pinky_list.append(len_pinky)

        # 얼굴 축 각도도 스텝 평균용으로 누적
        theta_horz_list.append(th_h)
        theta_vert_list.append(th_v)

        # ---- 기본 48D 피처 누적 ----
        feat_base = np.concatenate([hand_flat, face_aux], axis=0)
        feature_buffer.append(feat_base)

        # ---- 스텝이 차면 평균/다운샘플 ----
        if len(feature_buffer) == step:
            avg_base = np.mean(np.array(feature_buffer), axis=0)

            # 스텝 평균 각(원형 평균)
            th_mid   = mean_angle_toroidal(np.array(theta_mid_list,   dtype=np.float32)) if len(theta_mid_list)   > 0 else None
            th_pinky = mean_angle_toroidal(np.array(theta_pinky_list, dtype=np.float32)) if len(theta_pinky_list) > 0 else None
            th_horz  = mean_angle_toroidal(np.array(theta_horz_list,  dtype=np.float32)) if len(theta_horz_list)  > 0 else 0.0
            th_vert  = mean_angle_toroidal(np.array(theta_vert_list,  dtype=np.float32)) if len(theta_vert_list)  > 0 else 0.0

            # 중지(Middle) 각도 피처 (6D) + 길이(1)
            if th_mid is not None:
                d_h_m = wrap_angle(th_mid - th_horz)
                d_v_m = wrap_angle(th_mid - th_vert)
                mid_feats = np.array([
                    np.cos(th_mid),  np.sin(th_mid),
                    np.cos(d_h_m),   np.sin(d_h_m),
                    np.cos(d_v_m),   np.sin(d_v_m)
                ], dtype=np.float32)
                len_mid_step = float(np.mean(len_mid_list)) if len(len_mid_list) > 0 else 0.0
            else:
                mid_feats = np.zeros(6, dtype=np.float32)
                len_mid_step = 0.0

            # 새끼(Pinky) 각도 피처 (6D) + 길이(1)
            if th_pinky is not None:
                d_h_p = wrap_angle(th_pinky - th_horz)
                d_v_p = wrap_angle(th_pinky - th_vert)
                pinky_feats = np.array([
                    np.cos(th_pinky), np.sin(th_pinky),
                    np.cos(d_h_p),    np.sin(d_h_p),
                    np.cos(d_v_p),    np.sin(d_v_p)
                ], dtype=np.float32)
                len_pinky_step = float(np.mean(len_pinky_list)) if len(len_pinky_list) > 0 else 0.0
            else:
                pinky_feats = np.zeros(6, dtype=np.float32)
                len_pinky_step = 0.0

            # 최종 62D 피처 결합: 48 + 6 + 6 + 1 + 1
            extra = np.concatenate([
                mid_feats, pinky_feats,
                np.array([len_mid_step, len_pinky_step], dtype=np.float32)
            ], axis=0)
            feat62 = np.concatenate([avg_base, extra], axis=0)

            T.append(feat62)

            # 버퍼 초기화
            feature_buffer = []
            theta_mid_list.clear()
            theta_pinky_list.clear()
            theta_horz_list.clear()
            theta_vert_list.clear()
            len_mid_list.clear()
            len_pinky_list.clear()

    # 남은 버퍼 처리
    if feature_buffer:
        avg_base = np.mean(np.array(feature_buffer), axis=0)

        th_mid   = mean_angle_toroidal(np.array(theta_mid_list,   dtype=np.float32)) if len(theta_mid_list)   > 0 else None
        th_pinky = mean_angle_toroidal(np.array(theta_pinky_list, dtype=np.float32)) if len(theta_pinky_list) > 0 else None
        th_horz  = mean_angle_toroidal(np.array(theta_horz_list,  dtype=np.float32)) if len(theta_horz_list)  > 0 else 0.0
        th_vert  = mean_angle_toroidal(np.array(theta_vert_list,  dtype=np.float32)) if len(theta_vert_list)  > 0 else 0.0

        if th_mid is not None:
            d_h_m = wrap_angle(th_mid - th_horz)
            d_v_m = wrap_angle(th_mid - th_vert)
            mid_feats = np.array([
                np.cos(th_mid),  np.sin(th_mid),
                np.cos(d_h_m),   np.sin(d_h_m),
                np.cos(d_v_m),   np.sin(d_v_m)
            ], dtype=np.float32)
            len_mid_step = float(np.mean(len_mid_list)) if len(len_mid_list) > 0 else 0.0
        else:
            mid_feats = np.zeros(6, dtype=np.float32)
            len_mid_step = 0.0

        if th_pinky is not None:
            d_h_p = wrap_angle(th_pinky - th_horz)
            d_v_p = wrap_angle(th_pinky - th_vert)
            pinky_feats = np.array([
                np.cos(th_pinky), np.sin(th_pinky),
                np.cos(d_h_p),    np.sin(d_h_p),
                np.cos(d_v_p),    np.sin(d_v_p)
            ], dtype=np.float32)
            len_pinky_step = float(np.mean(len_pinky_list)) if len(len_pinky_list) > 0 else 0.0
        else:
            pinky_feats = np.zeros(6, dtype=np.float32)
            len_pinky_step = 0.0

        extra = np.concatenate([
            mid_feats, pinky_feats,
            np.array([len_mid_step, len_pinky_step], dtype=np.float32)
        ], axis=0)
        feat62 = np.concatenate([avg_base, extra], axis=0)
        T.append(feat62)

    cap.release()
    face_model.close()
    hand_model.close()

    if len(T) == 0:
        return np.zeros((0, FEAT_DIM), dtype=np.float32)
    return np.vstack(T)

# -------------------- 메인 실행 함수 --------------------
def main():
    ap = argparse.ArgumentParser(description="영상으로부터 손/얼굴 특징을 추출하는 스크립트")
    ap.add_argument('--video_root', default=None, help='원본 비디오가 담긴 video_data 루트')
    ap.add_argument('--project_tag', required=True, help='프로젝트 버전 태그 (예: v1_facehand_angles)')
    ap.add_argument('--add_eye_feats', action='store_true', default=False,
                    help='[deprecated/ignored] 예전 보조피처 플래그 (현재 버전에서 무시됨)')
    args = ap.parse_args()

    video_root = resolve_video_root(args.video_root)
    print(f"[INFO] Source video root = {video_root}")

    out_root = os.path.join(os.getcwd(), args.project_tag, f"data_{args.project_tag}")
    ensure_dir(out_root)
    print(f"[INFO] Output feature root = {out_root}")

    log = []
    t0 = time.time()
    n_ok, n_fail = 0, 0
    video_list = list(list_videos(video_root))
    print(f"[INFO] Found {len(video_list)} videos to process.")

    for i, (label, person, vpath) in enumerate(video_list):
        print(f"[INFO] Processing video {i+1}/{len(video_list)}: {vpath}")
        try:
            arr = extract_features_from_video(vpath, use_eye_feats=args.add_eye_feats)

            if arr.shape[0] == 0:
                print(f"⚠️ SKIP {vpath}: No valid frames found.")
                log.append({'label': label, 'person': person, 'video': vpath, 'error': 'No valid frames'})
                n_fail += 1
                continue

            save_dir = os.path.join(out_root, label, person)
            ensure_dir(save_dir)
            base = os.path.splitext(os.path.basename(vpath))[0]
            npy_path = os.path.join(save_dir, base + '.npy')
            np.save(npy_path, arr)

            log.append({'label': label, 'person': person, 'video': vpath,
                        'frames': int(arr.shape[0]),
                        'feat_dim': int(arr.shape[1]),
                        'target_fps': TARGET_FPS, 'size': f'{TARGET_W}x{TARGET_H}'})
            n_ok += 1
        except Exception as e:
            log.append({'label': label, 'person': person, 'video': vpath, 'error': str(e)})
            print(f"⚠️ FAIL {vpath}: {e}")
            n_fail += 1

    log_dir = os.path.join(os.getcwd(), 'logs')
    ensure_dir(log_dir)
    log_filepath = os.path.join(log_dir, f'{args.project_tag}_extraction_log.json')
    with open(log_filepath, 'w', encoding='utf-8') as f:
        json.dump({
            'project_tag': args.project_tag,
            'items': log,
            'sec': time.time() - t0,
            'ok': n_ok,
            'fail': n_fail
        }, f, ensure_ascii=False, indent=2)

    print(f'\nDone. Saved log to {log_filepath} (ok={n_ok}, fail={n_fail})')

if __name__ == '__main__':
    main()
