# run.py
import os
import json
import optuna
import logging
import copy
import hashlib
import tempfile
import shutil
import time
import numpy as np

import main as pipeline
import config
import m1_feature_extraction_answer as feature_extraction

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def _ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def _hash_feature_cfg(fe_cfg: dict) -> str:
    """
    특징 추출 설정을 JSON으로 안정 직렬화하여 해시를 생성.
    dict 순서 변화로 다른 파일을 만드는 일을 방지.
    """
    j = json.dumps(fe_cfg, ensure_ascii=False, sort_keys=True)
    return hashlib.md5(j.encode('utf-8')).hexdigest()[:8]


def get_or_create_features(cfg):
    """
    - 캐시가 있으면 즉시 재사용
    - 없으면 임시 디렉터리에서 생성 → 무결성 확인 → 원자적 이동
    - 동시 실행 대비 .lock 파일 사용
    """
    fe_cfg = cfg.get('feature_extraction', {})
    params_hash = _hash_feature_cfg(fe_cfg)

    feature_store_dir = cfg['paths'].get('FEATURE_STORE', os.path.join('outputs', 'feature_store'))
    _ensure_dir(feature_store_dir)
    features_path = os.path.join(feature_store_dir, f"features_{params_hash}.npz")
    lock_path = features_path + ".lock"

    # 이미 존재하면 재사용
    if os.path.exists(features_path):
        logging.info(f"[재사용] 특징 캐시: {os.path.basename(features_path)}")
        return features_path

    # 락 시도
    got_lock = False
    try:
        fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.close(fd)
        got_lock = True
    except FileExistsError:
        got_lock = False

    # 다른 프로세스가 만드는 중이면 기다렸다가 재사용
    if not got_lock:
        logging.info(f"[대기] 다른 프로세스가 특징 생성 중… ({os.path.basename(features_path)})")
        for _ in range(180):
            if os.path.exists(features_path):
                logging.info(f"[대기완료] 캐시 생성됨 → 재사용")
                return features_path
            time.sleep(1)
        logging.error("특징 생성 대기 시간 초과")
        # 타임아웃 시에도 안전하게 직접 생성 시도 (다시 락 시도)
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
            got_lock = True
        except FileExistsError:
            # 여전히 누군가 보유 → 실패 반환
            return None

    tmpdir = None
    try:
        logging.info(f"[생성] 캐시 없음 → 새로 생성: {os.path.basename(features_path)}")
        tmpdir = tempfile.mkdtemp(prefix=f"fe_{params_hash}_", dir=feature_store_dir)

        temp_cfg = copy.deepcopy(cfg)
        temp_cfg['paths']['OUTPUT_DIR'] = tmpdir  # m1이 여기 아래 processed_data/에 저장
        # RAW_LANDMARK_STORE가 없으면 기본값 유지
        temp_cfg['paths'].setdefault('RAW_LANDMARK_STORE',
                                     cfg['paths'].get('RAW_LANDMARK_STORE',
                                                      os.path.join('outputs', 'raw_landmark_store')))

        feature_extraction.main(temp_cfg)

        default_path = os.path.join(tmpdir, 'processed_data', 'features.npz')
        if not os.path.exists(default_path):
            raise FileNotFoundError(f"산출물 없음: {default_path}")

        # 무결성 검증 (파일 핸들 닫히도록 with 사용)
        with np.load(default_path, allow_pickle=True) as arr:
            _ = arr['X']; _ = arr['y']; _ = arr['groups']; _ = arr['class_names']

        # 원자적 이동
        os.replace(default_path, features_path)
        logging.info(f"[완료] 캐시 저장: {features_path}")
        return features_path

    except Exception:
        logging.error("특징 생성 중 오류", exc_info=True)
        try:
            if os.path.exists(features_path):
                os.remove(features_path)
        except Exception:
            pass
        return None
    finally:
        # 락/임시디렉토리 정리
        try:
            if os.path.exists(lock_path):
                os.remove(lock_path)
        except Exception:
            pass
        if tmpdir and os.path.exists(tmpdir):
            shutil.rmtree(tmpdir, ignore_errors=True)
        # 혹시 남아있을 processed_data 폴더 정리(실패해도 무시)
        try:
            pd_dir = os.path.join(feature_store_dir, 'processed_data')
            if os.path.isdir(pd_dir):
                shutil.rmtree(pd_dir, ignore_errors=True)
        except Exception:
            pass


def objective(trial: optuna.Trial):
    cfg = config.define_search_space(trial)
    logging.info(f"\n\n===== Optuna Trial {trial.number} 시작 | Params: {trial.params} =====")
    try:
        features_path = get_or_create_features(cfg)
        if not features_path:
            logging.error("특징 준비 실패 → 0.0 반환")
            return 0.0
        cfg['paths']['FEATURES_PATH'] = features_path
        avg_macro_f1 = pipeline.run_pipeline(cfg, trial)
        return 0.0 if avg_macro_f1 is None else float(avg_macro_f1)
    except optuna.exceptions.TrialPruned:
        logging.info(f"Trial {trial.number} pruned")
        raise
    except Exception:
        logging.error(f"Trial {trial.number} 실행 오류", exc_info=True)
        return 0.0


def main():
    if config.RUN_MODE == 'MANUAL':
        logging.info("===== MANUAL 모드 =====")
        cfg = copy.deepcopy(config.MANUAL_CONFIG)
        features_path = get_or_create_features(cfg)
        if not features_path:
            logging.error("특징 준비 실패로 종료")
            return
        cfg['paths']['FEATURES_PATH'] = features_path
        pipeline.run_pipeline(cfg, trial=None)

    elif config.RUN_MODE == 'OPTIMIZE':
        logging.info("===== OPTIMIZE 모드 시작 =====")
        study = optuna.create_study(
            direction='maximize',
            study_name='Brush-Action-Classification-HPO',
            pruner=optuna.pruners.MedianPruner()
        )
        try:
            # n_jobs 기본 1 (GPU 공유 → 병렬 비권장). 필요 시 config로 제어 가능.
            study.optimize(objective, n_trials=50)
        except KeyboardInterrupt:
            logging.warning("사용자 중단")
        logging.info("===== 최적화 완료 =====")

        if study.best_trial:
            bt = study.best_trial
            logging.info(f"최고 성능 Macro F1: {bt.value:.4f}")
            logging.info("최적 하이퍼파라미터:")
            for k, v in bt.params.items():
                logging.info(f"  - {k}: {v}")

            output_dir = 'outputs'; _ensure_dir(output_dir)
            try:
                df = study.trials_dataframe()
                csv_path = os.path.join(output_dir, 'optuna_results_summary.csv')
                df.to_csv(csv_path, index=False)
                logging.info(f"결과 CSV 저장: {csv_path}")
            except Exception:
                logging.warning("optuna dataframe 생성 실패", exc_info=True)

            # 간단 시각화는 옵션 (plotly 미설치 시 건너뜀)
            try:
                import optuna.visualization as vis
                fig1 = vis.plot_optimization_history(study)
                fig1.write_html(os.path.join(output_dir, 'optuna_history.html'))
                fig2 = vis.plot_param_importances(study)
                fig2.write_html(os.path.join(output_dir, 'optuna_importances.html'))
                logging.info("시각화 HTML 저장 완료")
            except Exception:
                logging.warning("Optuna 시각화 건너뜀", exc_info=True)
        else:
            logging.info("유효한 Trial 없음")

    else:
        logging.error("config.RUN_MODE를 'MANUAL' 또는 'OPTIMIZE'로 설정하세요.")


if __name__ == '__main__':
    main()
