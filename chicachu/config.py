import os
import optuna

# =========================
# 실행 모드
# =========================
RUN_MODE = 'MANUAL'   # 'MANUAL' | 'OPTIMIZE'
GLOBAL_SEED = 42

def _default_workers() -> int:
    return max(1, (os.cpu_count() or 2) - 1)

# =========================
# 1) MANUAL
# =========================
MANUAL_CONFIG = {
    'paths': {
        'VIDEO_ROOT': 'video_data',
        'OUTPUT_DIR': 'outputs/manual_run',
        'FEATURE_STORE': 'outputs/feature_store',
        'RAW_LANDMARK_STORE': 'outputs/raw_landmark_store',
    },

    'feature_extraction': {
        'FEATURE_CONFIG': {
            'hand_all': True,
            'face_roll_scale': True,
            'face_pitch': True,
            'face_mouth_corners': True,
            'face_inner_eyes': False,
        },
        'WINDOW_SIZE': 30,
        'STRIDE': 3,

        'USE_RAW_LANDMARK_CACHE': True,
        'INTERPOLATE_MISSING': True,
        'SMOOTHING': {
            'enabled': True,
            'method': 'savgol',
            'window_length': 7,
            'polyorder': 2,
            'ema_alpha': 0.2,
        },
        'TEMPORAL_FEATURES': {
            'velocity': True,
            'acceleration': False,
            'segment_stats': False,
        },

        'PARALLEL': True,
        'NUM_WORKERS': 8,
    },

    # ▶ 기본 아키텍처(아발론2 결과 반영: CNN_GRU 권장)
    'model_arch': {
        'variant': 'cnn_gru',   # 'gru_attn' | 'gru' | 'bigru' | 'gru_sa' | 'cnn_gru'
        'gru1_units': 160,
        'gru2_units': 0,
        'dense_units': 32,
        'dropout_rate': 0.377332,
        'recurrent_dropout': 0.0,   # CuDNN 가속 위해 0 유지 권장
        'use_layer_norm': True,
        'l2': 5.6e-05,
        'conv_frontend': {
            'enabled': True,
            'filters': 48,
            'kernel_size': 3,
            'strides': 1,
            'pool_size': 3,
            'separable': True,
            'dropout': 0.1,
            'l2': 4.0e-05,
            'use_ln': True
        },
        'self_attention': {  # variant='gru_sa'일 때만 사용
            'num_heads': 4, 'key_dim': 32, 'dropout': 0.1
        }
    },

    'training': {
        'learning_rate': 3.57e-04,
        'batch_size': 64,
        'epochs': 100,
        'early_stopping_patience': 10,
    },

    'settings': {
        'SEED': GLOBAL_SEED,
        'VAL_SPLIT_GROUP_BY': 'video',   # 'video' | 'window'
        'VALIDATION_SPLIT_FROM_TRAIN': 0.15,
        'SAVE_VISUALIZATIONS': True,
        'EVAL': {
            'bootstrap_ci': False,
            'n_bootstrap': 1000,
            'roc_ova': False,
            'calibration': False,
            'save_raw_predictions': True,
        'FULL_TRAIN': True,  # 전체 데이터 단일 학습 모드
        },
    }
}

# =========================
# 2) OPTIMIZE
# =========================
def define_search_space(trial: optuna.Trial):
    cfg = {
        'paths': {
            'VIDEO_ROOT': 'video_data',
            'OUTPUT_DIR': f'outputs/trial_{trial.number}',
            'FEATURE_STORE': 'outputs/feature_store',
            'RAW_LANDMARK_STORE': 'outputs/raw_landmark_store',
        },
        'feature_extraction': {
            'FEATURE_CONFIG': {
                'hand_all': True,
                'face_roll_scale': True,
                'face_pitch': True,
                'face_mouth_corners': False,
                'face_inner_eyes': False,
            },
            'WINDOW_SIZE': trial.suggest_int('WINDOW_SIZE', 15, 45, step=15),
            'STRIDE': trial.suggest_int('STRIDE', 1, 5, step=2),

            'USE_RAW_LANDMARK_CACHE': True,
            'INTERPOLATE_MISSING': True,
            'SMOOTHING': {
                'enabled': False,
                'method': 'savgol',
                'window_length': 7,
                'polyorder': 2,
                'ema_alpha': 0.2,
            },
            'TEMPORAL_FEATURES': {
                'velocity': True,
                'acceleration': True,
                'segment_stats': True,
            },
            'PARALLEL': True,
            'NUM_WORKERS': _default_workers(),
        },
        'model_arch': {
            # 필요하면 variant도 탐색 가능(여기선 구조 고정)
            'variant': 'cnn_gru',
            'gru1_units': trial.suggest_categorical('gru1_units', [96, 128, 160, 256]),
            'gru2_units': trial.suggest_categorical('gru2_units', [0, 64, 96]),
            'dense_units': trial.suggest_categorical('dense_units', [32, 64, 96]),
            'dropout_rate': trial.suggest_float('dropout_rate', 0.3, 0.6),
            'recurrent_dropout': 0.0,
            'use_layer_norm': True,
            'l2': trial.suggest_float('l2', 1e-5, 3e-4, log=True),
            'conv_frontend': {
                'enabled': True,
                'filters': trial.suggest_categorical('cf_filters', [48, 64, 80]),
                'kernel_size': trial.suggest_categorical('cf_kernel', [3, 5, 7]),
                'strides': 1,
                'pool_size': trial.suggest_categorical('cf_pool', [2, 3]),
                'separable': True,
                'dropout': trial.suggest_categorical('cf_drop', [0.0, 0.1, 0.2]),
                'l2': trial.suggest_float('cf_l2', 1e-5, 3e-4, log=True),
                'use_ln': True
            }
        },
        'training': {
            'learning_rate': trial.suggest_float('learning_rate', 1e-4, 7e-4, log=True),
            'batch_size': trial.suggest_categorical('batch_size', [64, 96, 128]),
            'epochs': 100,
            'early_stopping_patience': 5,
        },
        'settings': {
            'SEED': GLOBAL_SEED,
            'VAL_SPLIT_GROUP_BY': 'video',
            'VALIDATION_SPLIT_FROM_TRAIN': 0.15,
            'SAVE_VISUALIZATIONS': False,
            'EVAL': {
                'bootstrap_ci': False,
                'n_bootstrap': 1000,
                'roc_ova': False,
                'calibration': False,
                'save_raw_predictions': True,
            },
        }
    }
    return cfg
