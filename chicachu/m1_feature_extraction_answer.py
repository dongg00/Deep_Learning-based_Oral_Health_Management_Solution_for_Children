# m1_feature_extraction_answer.py
# 특징 추출 + 정규화 + 윈도우링 + (옵션) 캐시 + 멀티프로세싱 병렬화 (BrokenProcessPool 자동 폴백 포함)

import os
import cv2
import math
import json
import time
import shutil
import tempfile
import logging
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed, BrokenProcessPool

import mediapipe as mp
import utils

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

mp_hands = mp.solutions.hands
mp_face_mesh = mp.solutions.face_mesh

# FaceMesh 인덱스 매핑(사용자 정의)
LANDMARK_MAP = {
    'face_roll_scale': [33, 263],      # 양쪽 눈 바깥
    'face_pitch': [1, 152],            # 코, 턱
    'face_mouth_corners': [61, 291],   # 입꼬리
    'face_inner_eyes': [133, 362],     # 눈 안쪽
}

NUM_HAND_LMS = 21
NUM_FACE_LMS = 468  # 전체 저장(캐시 재사용성↑)

# ------------------------------- 공통 유틸 -------------------------------
def _ensure_dir(p): os.makedirs(p, exist_ok=True)

def get_face_indices_from_config(cfg):
    indices = []
    feat_cfg = cfg['feature_extraction']['FEATURE_CONFIG']
    for key, on in feat_cfg.items():
        if on and key.startswith('face'):
            indices.extend(LANDMARK_MAP.get(key, []))
    final = list(dict.fromkeys(indices))
    logging.info(f"Config 기반 얼굴 랜드마크 인덱스(선택): {final}")
    return final

def interpolate_nan_along_time(arr):
    # arr: (T, K, 2)
    T, K, C = arr.shape
    out = arr.copy()
    for k in range(K):
        for c in range(C):
            series = out[:, k, c]
            idx = np.arange(T)
            mask = ~np.isnan(series)
            if mask.any():
                first, last = np.argmax(mask), T - 1 - np.argmax(mask[::-1])
                series[:first] = series[first]
                series[last+1:] = series[last]
                mask = ~np.isnan(series)
                if mask.sum() >= 2:
                    out[:, k, c] = np.interp(idx, idx[mask], series[mask])
    return out

def smooth_timeseries(arr, method='savgol', window_length=7, polyorder=2, ema_alpha=0.2):
    T, K, C = arr.shape
    out = arr.copy()
    try:
        if method == 'savgol':
            from scipy.signal import savgol_filter
            wl = max(3, window_length | 1)  # 홀수
            for k in range(K):
                for c in range(C):
                    out[:, k, c] = savgol_filter(out[:, k, c], window_length=wl, polyorder=polyorder, mode='interp')
        elif method == 'ema':
            for k in range(K):
                for c in range(C):
                    x = out[:, k, c]
                    for t in range(1, T):
                        x[t] = ema_alpha * x[t] + (1 - ema_alpha) * x[t-1]
                    out[:, k, c] = x
    except Exception:
        pass
    return out

def normalize_sequence(raw, cfg, face_indices):
    """
    raw: {'hand':(T,21,2), 'face_all':(T,468,2)}
    face_indices: 사용할 얼굴 인덱스 리스트(정규화/특징 선택에도 사용)
    """
    feat_cfg = cfg['feature_extraction']['FEATURE_CONFIG']
    T = raw['hand'].shape[0]

    # 필요한 얼굴만 골라내기
    if face_indices:
        face = raw['face_all'][:, face_indices, :]  # (T, Fsel, 2)
        idx_map = {orig_idx: i for i, orig_idx in enumerate(face_indices)}
    else:
        face = np.zeros((T, 0, 2), dtype=np.float32)
        idx_map = {}

    hand = raw['hand'].copy()

    # 위치 원점: 코(1)
    if feat_cfg.get('face_pitch') and 1 in idx_map:
        origin = face[:, idx_map[1], :]  # (T,2)
    else:
        origin = np.zeros((T, 2), dtype=np.float32)

    hand = hand - origin[:, None, :]
    if face.shape[1] > 0:
        face = face - origin[:, None, :]

    # 크기/롤 보정: 좌/우 눈 바깥(33/263)
    if feat_cfg.get('face_roll_scale') and 33 in idx_map and 263 in idx_map:
        left_eye = face[:, idx_map[33], :]
        right_eye = face[:, idx_map[263], :]
        scale = np.linalg.norm(right_eye - left_eye, axis=1)  # (T,)
        scale[scale < 1e-6] = 1.0
        hand = hand / scale[:, None, None]
        if face.shape[1] > 0:
            face = face / scale[:, None, None]

        # 롤 회전
        angles = np.arctan2(right_eye[:, 1] - left_eye[:, 1], right_eye[:, 0] - left_eye[:, 0])  # (T,)
        cos_a, sin_a = np.cos(-angles), np.sin(-angles)
        R = np.stack([np.stack([cos_a, -sin_a], axis=1),
                      np.stack([sin_a,  cos_a], axis=1)], axis=1)   # (T,2,2)
        hand = np.einsum('tij,tkj->tki', R, hand)
        if face.shape[1] > 0:
            face = np.einsum('tij,tkj->tki', R, face)

    # 피치 보정: 코(1)->턱(152) 수직 정렬
    if feat_cfg.get('face_pitch') and 1 in idx_map and 152 in idx_map:
        nose = face[:, idx_map[1], :]
        chin = face[:, idx_map[152], :]
        angles = np.arctan2(chin[:, 1] - nose[:, 1], chin[:, 0] - nose[:, 0]) - (np.pi / 2)
        cos_a, sin_a = np.cos(-angles), np.sin(-angles)
        R = np.stack([np.stack([cos_a, -sin_a], axis=1),
                      np.stack([sin_a,  cos_a], axis=1)], axis=1)
        hand = np.einsum('tij,tkj->tki', R, hand)
        if face.shape[1] > 0:
            face = np.einsum('tij,tkj->tki', R, face)

    parts = []
    if feat_cfg.get('hand_all'):
        parts.append(hand.reshape(T, -1))
    if any(k.startswith('face') and v for k, v in feat_cfg.items()) and face.size > 0:
        parts.append(face.reshape(T, -1))

    if parts:
        seq = np.concatenate(parts, axis=1)  # (T, D)
    else:
        seq = hand.reshape(T, -1)  # 최소 손은 포함한다고 가정

    return seq  # (T, D)

def add_temporal_features(seq, cfg):
    tf_cfg = cfg['feature_extraction']['TEMPORAL_FEATURES']
    out = [seq]
    if tf_cfg.get('velocity'):
        vel = np.diff(seq, axis=0, prepend=seq[:1])
        out.append(vel)
    if tf_cfg.get('acceleration'):
        acc = np.diff(seq, axis=0, prepend=seq[:1])
        acc = np.diff(acc, axis=0, prepend=acc[:1])
        out.append(acc)
    return np.concatenate(out, axis=1)

def make_windows(seq, window, stride, add_segment_stats=False):
    windows = []
    metas = []
    T = seq.shape[0]
    for s in range(0, max(0, T - window + 1), stride):
        seg = seq[s:s + window]
        if seg.shape[0] < window:
            continue
        if add_segment_stats:
            mean = seg.mean(axis=0, keepdims=True)
            std = seg.std(axis=0, keepdims=True)
            seg = np.concatenate([seg, np.repeat(mean, window, axis=0), np.repeat(std, window, axis=0)], axis=1)
        windows.append(seg)
        metas.append((s, s + window))
    return windows, metas

# ---------------------------- 원시 추출/캐시 ----------------------------
def extract_raw_landmarks_single(video_path, mp_config):
    """
    mp_config: dict for mediapipe options
      {
        'hands': {'max_num_hands':1, 'model_complexity':0, ...},
        'face':  {'max_num_faces':1, 'refine_landmarks':False, ...}
      }
    """
    # 워커 내부에서 스레드 과사용 방지
    try:
        cv2.setNumThreads(1)
    except Exception:
        pass
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logging.error(f"비디오를 열 수 없습니다: {video_path}")
        return None

    # Mediapipe 솔루션은 프로세스/스레드 안전하지 않으므로 워커 내부에서 생성/폐기
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=mp_config.get('hands', {}).get('max_num_hands', 1),
        model_complexity=mp_config.get('hands', {}).get('model_complexity', 0),
        min_detection_confidence=mp_config.get('hands', {}).get('min_detection_confidence', 0.5),
        min_tracking_confidence=mp_config.get('hands', {}).get('min_tracking_confidence', 0.5),
    )
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=mp_config.get('face', {}).get('max_num_faces', 1),
        refine_landmarks=mp_config.get('face', {}).get('refine_landmarks', False),
        min_detection_confidence=mp_config.get('face', {}).get('min_detection_confidence', 0.5),
        min_tracking_confidence=mp_config.get('face', {}).get('min_tracking_confidence', 0.5),
    )

    hand_seq = []
    face_seq = []
    while True:
        ret, frame = cap.read()
        if not ret: break
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        hr = hands.process(image)
        fr = face_mesh.process(image)

        if hr.multi_hand_landmarks and fr.multi_face_landmarks:
            hand = np.array([[lm.x, lm.y] for lm in hr.multi_hand_landmarks[0].landmark], dtype=np.float32)  # (21,2)
            face = np.array([[lm.x, lm.y] for lm in fr.multi_face_landmarks[0].landmark], dtype=np.float32)  # (468,2)
        else:
            hand = np.full((NUM_HAND_LMS, 2), np.nan, dtype=np.float32)
            face = np.full((NUM_FACE_LMS, 2), np.nan, dtype=np.float32)

        hand_seq.append(hand)
        face_seq.append(face)

    cap.release()
    hands.close(); face_mesh.close()

    if not hand_seq:
        return None
    return {'hand': np.stack(hand_seq, axis=0), 'face_all': np.stack(face_seq, axis=0)}

def load_or_build_raw_cache(video_path, raw_store_dir, use_cache, mp_config):
    _ensure_dir(raw_store_dir)
    base = os.path.splitext(os.path.basename(video_path))[0]
    cache_path = os.path.join(raw_store_dir, f"{base}.npz")

    if use_cache and os.path.exists(cache_path):
        try:
            with np.load(cache_path) as d:
                return {'hand': d['hand'], 'face_all': d['face_all']}
        except Exception:
            logging.warning(f"원시 캐시 로드 실패 → 재생성: {cache_path}")

    raw = extract_raw_landmarks_single(video_path, mp_config)
    if raw is None:
        return None

    if use_cache:
        try:
            np.savez_compressed(cache_path, hand=raw['hand'], face_all=raw['face_all'])
        except Exception:
            logging.warning(f"원시 캐시 저장 실패: {cache_path}")

    return raw

# ----------------------------- 워커 함수 -----------------------------
def _worker_process(job):
    """
    한 비디오를 처리하여 임시 청크 파일(npz)을 생성하고, 경로/카운트 리턴.
    job: dict with keys:
      - video_path, class_name, person_id, label, group, cfg_json, face_indices, tmp_chunk_dir
    """
    try:
        # 안전하게 딕셔너리 복원
        cfg = json.loads(job['cfg_json'])
        face_indices = job['face_indices']

        # Mediapipe 빠른 옵션 (필요 시 config에 노출 가능)
        mp_config = {
            'hands': {
                'max_num_hands': 1,
                'model_complexity': 0,  # 0이 가장 빠름
                'min_detection_confidence': 0.5,
                'min_tracking_confidence': 0.5,
            },
            'face': {
                'max_num_faces': 1,
                'refine_landmarks': False,  # 속도↑
                'min_detection_confidence': 0.5,
                'min_tracking_confidence': 0.5,
            }
        }

        raw = load_or_build_raw_cache(
            video_path=job['video_path'],
            raw_store_dir=cfg['paths'].get('RAW_LANDMARK_STORE', os.path.join('outputs', 'raw_landmark_store')),
            use_cache=cfg['feature_extraction'].get('USE_RAW_LANDMARK_CACHE', False),
            mp_config=mp_config
        )
        if raw is None:
            return {'ok': False, 'reason': 'raw_none'}

        # 옵션: 결측 보간/스무딩
        if cfg['feature_extraction'].get('INTERPOLATE_MISSING', True):
            raw['hand'] = interpolate_nan_along_time(raw['hand'])
            raw['face_all'] = interpolate_nan_along_time(raw['face_all'])

        sm = cfg['feature_extraction'].get('SMOOTHING', {})
        if sm.get('enabled', False):
            raw['hand'] = smooth_timeseries(raw['hand'], method=sm.get('method', 'savgol'),
                                            window_length=sm.get('window_length', 7),
                                            polyorder=sm.get('polyorder', 2),
                                            ema_alpha=sm.get('ema_alpha', 0.2))
            raw['face_all'] = smooth_timeseries(raw['face_all'], method=sm.get('method', 'savgol'),
                                                window_length=sm.get('window_length', 7),
                                                polyorder=sm.get('polyorder', 2),
                                                ema_alpha=sm.get('ema_alpha', 0.2))

        seq = normalize_sequence(raw, cfg, face_indices)
        seq = add_temporal_features(seq, cfg)

        window = cfg['feature_extraction']['WINDOW_SIZE']
        stride = cfg['feature_extraction']['STRIDE']
        add_seg_stats = cfg['feature_extraction'].get('TEMPORAL_FEATURES', {}).get('segment_stats', False)

        windows, metas = make_windows(seq, window, stride, add_segment_stats=add_seg_stats)
        if not windows:
            return {'ok': False, 'reason': 'no_windows'}

        Xv = np.asarray(windows, dtype=np.float32)
        yv = np.full((len(windows),), job['label'], dtype=np.int32)
        gv = np.full((len(windows),), job['group'], dtype=np.int32)
        kv = np.array([job['key']] * len(windows))

        # 임시 청크 파일로 저장
        _ensure_dir(job['tmp_chunk_dir'])
        chunk_path = os.path.join(job['tmp_chunk_dir'], f"{job['key_hash']}.npz")
        np.savez_compressed(chunk_path, X=Xv, y=yv, g=gv, k=kv)

        return {'ok': True, 'chunk_path': chunk_path, 'n': len(windows)}
    except Exception as e:
        return {'ok': False, 'reason': f'exception: {e}'}

# ------------------------------- 메인 -------------------------------
def hashlib_md5_hex(s: str) -> str:
    import hashlib
    return hashlib.md5(s.encode('utf-8')).hexdigest()[:16]

def main(cfg):
    logging.info("===== 1. 특징 추출 프로세스 시작 =====")
    video_root = cfg['paths']['VIDEO_ROOT']
    output_dir = cfg['paths']['OUTPUT_DIR']
    raw_store = cfg['paths'].get('RAW_LANDMARK_STORE', os.path.join('outputs', 'raw_landmark_store'))

    window = cfg['feature_extraction']['WINDOW_SIZE']
    stride = cfg['feature_extraction']['STRIDE']

    # 병렬 처리 설정
    num_workers = int(cfg['feature_extraction'].get('NUM_WORKERS', max(1, (os.cpu_count() or 2) - 1)))
    parallel = bool(cfg['feature_extraction'].get('PARALLEL', True))
    logging.info(f"[병렬] PARALLEL={parallel}, NUM_WORKERS={num_workers}")

    face_indices = get_face_indices_from_config(cfg)

    # 클래스/사람/비디오 스캔
    class_names = sorted([d for d in os.listdir(video_root) if os.path.isdir(os.path.join(video_root, d))])
    class_to_label = {name: i for i, name in enumerate(class_names)}
    person_id_list = sorted(list(set(
        p for c in class_names
        for p in os.listdir(os.path.join(video_root, c))
        if os.path.isdir(os.path.join(video_root, c, p))
    )))
    person_to_group = {p: i for i, p in enumerate(person_id_list)}
    video_list = [
        (c, pid, vf)
        for c in class_names
        for pid in os.listdir(os.path.join(video_root, c))
        if os.path.isdir(os.path.join(video_root, c, pid))
        for vf in os.listdir(os.path.join(video_root, c, pid))
        if vf.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))
    ]
    logging.info(f"[진단] 클래스 폴더 수: {len(class_names)} → {class_names[:8]} ...")
    logging.info(f"[진단] 비디오 파일 수: {len(video_list)}")

    # 임시 청크 디렉토리
    tmp_chunk_dir = os.path.join(output_dir, '_tmp_chunks')
    _ensure_dir(tmp_chunk_dir)

    jobs = []
    for (class_name, person_id, video_file) in video_list:
        video_path = os.path.join(video_root, class_name, person_id, video_file)
        label = class_to_label[class_name]
        group = person_to_group[person_id]
        key = f"{class_name}/{person_id}/{os.path.splitext(video_file)[0]}"
        key_hash = hashlib_md5_hex(key)

        jobs.append({
            'video_path': video_path,
            'class_name': class_name,
            'person_id': person_id,
            'label': int(label),
            'group': int(group),
            'key': key,
            'key_hash': key_hash,
            'cfg_json': json.dumps(cfg, ensure_ascii=False),
            'face_indices': face_indices,
            'tmp_chunk_dir': tmp_chunk_dir
        })

    # 실행 (병렬 → 실패 시 순차 폴백)
    def _run_sequential(jobs_):
        chunk_paths_, total_windows_ = [], 0
        for job in tqdm(jobs_, desc="특징 추출(순차)"):
            res = _worker_process(job)
            if res.get('ok'):
                chunk_paths_.append(res['chunk_path'])
                total_windows_ += res.get('n', 0)
            else:
                logging.warning(f"워커 실패: {res.get('reason')}")
        return chunk_paths_, total_windows_

    chunk_paths = []
    total_windows = 0

    if parallel and num_workers > 1:
        try:
            with ProcessPoolExecutor(max_workers=num_workers) as ex:
                futures = [ex.submit(_worker_process, job) for job in jobs]
                for f in tqdm(as_completed(futures), total=len(futures), desc="특징 추출(병렬)"):
                    res = f.result()
                    if res.get('ok'):
                        chunk_paths.append(res['chunk_path'])
                        total_windows += res.get('n', 0)
                    else:
                        logging.warning(f"워커 실패: {res.get('reason')}")
        except BrokenProcessPool as e:
            logging.warning(f"[폴백] BrokenProcessPool 감지 → 순차 처리로 재시도: {e}")
            chunk_paths, total_windows = _run_sequential(jobs)
        except Exception as e:
            logging.warning(f"[폴백] 병렬 처리 예외 → 순차 처리로 재시도: {e}")
            chunk_paths, total_windows = _run_sequential(jobs)
    else:
        chunk_paths, total_windows = _run_sequential(jobs)

    if not chunk_paths:
        logging.error("생성된 청크가 없습니다. 종료.")
        return

    # -------------------- 청크 합치기 --------------------
    output_data_dir = os.path.join(output_dir, 'processed_data')
    _ensure_dir(output_data_dir)
    output_file_path = os.path.join(output_data_dir, 'features.npz')

    X_list, y_list, g_list, k_list = [], [], [], []
    for cp in tqdm(chunk_paths, desc="청크 병합"):
        try:
            with np.load(cp, allow_pickle=True) as d:
                X_list.append(d['X'])
                y_list.append(d['y'])
                g_list.append(d['g'])
                k_list.append(d['k'])
        except Exception:
            logging.warning(f"청크 로드 실패: {cp}")

    if not X_list:
        logging.error("병합할 데이터가 없습니다.")
        return

    X_final = np.concatenate(X_list, axis=0).astype(np.float32)
    y_final = np.concatenate(y_list, axis=0).astype(np.int32)
    g_final = np.concatenate(g_list, axis=0).astype(np.int32)
    meta_keys = np.concatenate(k_list, axis=0)

    np.savez_compressed(
        output_file_path,
        X=X_final,
        y=y_final,
        groups=g_final,
        class_names=np.array(class_names),
        meta_video_keys=meta_keys,  # 검증용 메타
    )

    # 임시 청크 정리
    try:
        shutil.rmtree(tmp_chunk_dir, ignore_errors=True)
    except Exception:
        pass

    if cfg['settings']['SAVE_VISUALIZATIONS']:
        logging.info("시각화 결과물 생성 중...")
        dist_path = os.path.join(output_data_dir, 'class_distribution.png')
        utils.plot_class_distribution(y_final, class_names, dist_path)
        if len(X_final) > 0:
            sample_path = os.path.join(output_data_dir, 'normalized_sample_example.png')
            utils.plot_sample_landmarks(X_final[0], sample_path)
        logging.info(f"데이터 시각화 결과가 {output_data_dir}에 저장되었습니다.")

    logging.info(f"===== 특징 추출 완료. 결과: {output_file_path} / X shape={X_final.shape}, 총 윈도우={total_windows} =====")

if __name__ == '__main__':
    logging.warning("이 스크립트는 단독 실행용이 아닙니다. run.py/ablation에서 호출하세요.")