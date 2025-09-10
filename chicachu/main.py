# main.py
import os
import random
import numpy as np
import pandas as pd
import logging
from sklearn.model_selection import train_test_split
import optuna
import tensorflow as tf

import utils
import m2_train_answer as train
import m3_evaluate_answer as evaluate

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def _set_global_seed(seed: int):
    try:
        random.seed(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)
        logging.info(f"[SEED] Global seed set to {seed}")
    except Exception as e:
        logging.warning(f"[SEED] Failed to set seed: {e}")


def _split_train_val_by_video(train_indices, y_full, video_keys, val_ratio, seed):
    """
    train_indices(전체 인덱스) 중에서 video_key 단위로 val을 분리.
    stratify는 비디오별 대표 라벨(최빈값) 기반. (클래스 수 부족 시 폴백)
    """
    tr_vids = video_keys[train_indices]
    uniq_videos = np.unique(tr_vids)

    # 비디오별 대표 라벨(최빈값)
    vid_to_label = {}
    for v in uniq_videos:
        idxs = train_indices[tr_vids == v]
        labels = y_full[idxs]
        if len(labels) == 0:
            vid_to_label[v] = 0
        else:
            vals, cnts = np.unique(labels, return_counts=True)
            vid_to_label[v] = int(vals[np.argmax(cnts)])

    vids = np.array(list(uniq_videos))
    strat = np.array([vid_to_label[v] for v in vids])

    # stratify 가능한지 체크(각 클래스가 최소 2개 이상 비디오를 가져야 함)
    ok_stratify = True
    for cls, cnt in zip(*np.unique(strat, return_counts=True)):
        if cnt < 2:
            ok_stratify = False
            break

    try:
        if ok_stratify:
            vids_tr, vids_va = train_test_split(
                vids, test_size=val_ratio, random_state=seed, stratify=strat
            )
        else:
            logging.warning("[VAL SPLIT] Not enough videos per class for stratify → fallback to random split.")
            vids_tr, vids_va = train_test_split(vids, test_size=val_ratio, random_state=seed, shuffle=True)
    except ValueError:
        logging.warning("[VAL SPLIT] Stratified split failed → fallback to random split.")
        vids_tr, vids_va = train_test_split(vids, test_size=val_ratio, random_state=seed, shuffle=True)

    vids_tr_set, vids_va_set = set(vids_tr), set(vids_va)

    # 전체 인덱스로 환원
    tr_mask = np.array([vk in vids_tr_set for vk in tr_vids])
    va_mask = np.array([vk in vids_va_set for vk in tr_vids])

    tr_global_idx = train_indices[tr_mask]
    va_global_idx = train_indices[va_mask]
    return tr_global_idx, va_global_idx, vids_tr_set, vids_va_set


def run_pipeline(cfg, trial: optuna.Trial = None):
    """
    하나의 설정으로 LOSO 파이프라인 실행.
    cfg['paths']['FEATURES_PATH']가 이미 설정되어 있어야 함(run.py 혹은 run_ablation.py에서 세팅).
    """
    # 재현성
    _set_global_seed(cfg['settings'].get('SEED', 42))

    features_path = cfg['paths']['FEATURES_PATH']
    try:
        with np.load(features_path, allow_pickle=True) as data:
            X = data['X']
            y = data['y']
            groups = data['groups']
            class_names = data['class_names']
            video_keys = data.get('meta_video_keys', None)
    except Exception:
        logging.error(f"특징 파일 로드 실패: {features_path}", exc_info=True)
        return None

    n_classes = len(class_names)
    unique_groups = np.sort(np.unique(groups))  # ★ fold 순서 고정
    num_folds = len(unique_groups)

    all_fold_reports = []
    all_fold_histories = []

    for i, test_group_id in enumerate(unique_groups):
        fold = i + 1
        logging.info(f"===== Fold {fold}/{num_folds} 시작 (Test Subject ID: {test_group_id}) =====")

        test_indices = np.where(groups == test_group_id)[0]
        train_indices = np.where(groups != test_group_id)[0]

        y_train_full = y[train_indices]

    # 검증 분할
    val_ratio = cfg['settings']['VALIDATION_SPLIT_FROM_TRAIN']
    seed = cfg['settings'].get('SEED', 42)
    group_by = cfg['settings'].get('VAL_SPLIT_GROUP_BY', 'video')

    # ✅ 추가: 검증 비율이 0.0 이하이면 검증 완전 생략
    if val_ratio is not None and float(val_ratio) <= 0.0:
        tr_idx = train_indices
        va_idx = np.array([], dtype=int)
        vids_tr_set = vids_va_set = set()
    else:
        if group_by == 'video' and video_keys is not None:
            tr_idx, va_idx, vids_tr_set, vids_va_set = _split_train_val_by_video(
                train_indices, y, video_keys, val_ratio, seed
            )
        else:
            try:
                tr_idx, va_idx = train_test_split(
                    train_indices, test_size=val_ratio, random_state=seed, stratify=y_train_full
                )
            except ValueError:
                logging.warning("[VAL SPLIT] Window-level stratify failed → fallback to random split.")
                tr_idx, va_idx = train_test_split(
                    train_indices, test_size=val_ratio, random_state=seed, shuffle=True
                )
            vids_tr_set = vids_va_set = set()

        # 데이터 추출
        X_train, y_train = X[tr_idx], y[tr_idx]
        if va_idx.size == 0:
            X_val, y_val = None, None
        else:
            X_val, y_val = X[va_idx], y[va_idx]
        X_test, y_test = X[test_indices], y[test_indices]

        # ✅ 격리 검증 로그 (영상 키 중복 0 보장)
        try:
            if (video_keys is not None) and (group_by == 'video') and (va_idx.size > 0):
                tr_v = set(video_keys[tr_idx])
                va_v = set(video_keys[va_idx])
                te_v = set(video_keys[test_indices])
                inter_tr_va = tr_v & va_v
                inter_tr_te = tr_v & te_v
                inter_va_te = va_v & te_v
                logging.info(f"[CHECK] videos: train={len(tr_v)}, val={len(va_v)}, test={len(te_v)}")
                logging.info(f"[CHECK] overlap(train,val)={len(inter_tr_va)}, (train,test)={len(inter_tr_te)}, (val,test)={len(inter_va_te)}")
                assert len(inter_tr_va) == 0, "train과 val 사이에 같은 video가 섞였습니다!"
                assert len(inter_tr_te) == 0 and len(inter_va_te) == 0, "train/val과 test 사이에 같은 video가 섞였습니다!"
        except Exception as e:
            logging.warning(f"[CHECK] video 단위 격리 검증을 건너뜁니다: {e}")

        # 폴더/모델 경로
        fold_model_dir = os.path.join(cfg['paths']['OUTPUT_DIR'], 'models', f'fold_{fold}')
        os.makedirs(fold_model_dir, exist_ok=True)
        model_save_path = os.path.join(fold_model_dir, 'best_model.h5')  # 내부에서 .keras로 저장됨

        # (선택) 분할 목록 저장
        if video_keys is not None:
            def _dump_list(path, keys):
                with open(path, 'w', encoding='utf-8') as f:
                    for k in sorted(keys):
                        f.write(str(k) + '\n')
            # train/val/test 키 기록 (val이 없으면 val 파일은 비울 수도)
            _dump_list(os.path.join(fold_model_dir, 'train_videos.txt'), set(video_keys[tr_idx]))
            if va_idx.size > 0:
                _dump_list(os.path.join(fold_model_dir, 'val_videos.txt'),   set(video_keys[va_idx]))
            else:
                _dump_list(os.path.join(fold_model_dir, 'val_videos.txt'),   set())
            _dump_list(os.path.join(fold_model_dir, 'test_videos.txt'),  set(video_keys[test_indices]))

        # 학습
        train.main(X_train, y_train, X_val, y_val, n_classes, model_save_path, cfg)

        # 학습 기록 취합
        history_csv_path = os.path.join(fold_model_dir, 'history.csv')
        if os.path.exists(history_csv_path):
            history_df = pd.read_csv(history_csv_path)
            history_df['fold'] = fold
            all_fold_histories.append(history_df)

        # 평가(+ meta_keys 전달)
        test_meta = video_keys[test_indices] if video_keys is not None else None
        report_df = evaluate.main(X_test, y_test, class_names, model_save_path, fold_model_dir, cfg, meta_keys=test_meta)
        if report_df is not None:
            report_df['fold'] = fold
            all_fold_reports.append(report_df)
        else:
            logging.error(f"Fold {fold} 평가 실패. 이 Trial을 중단합니다.")
            # 세션 정리 후 가지치기
            tf.keras.backend.clear_session()
            raise optuna.exceptions.TrialPruned()

        # Optuna 가지치기
        if trial:
            intermediate_value = float(report_df.loc['macro avg', 'f1-score'])
            trial.report(intermediate_value, step=fold)
            if trial.should_prune():
                logging.info(f"Trial {trial.number} Fold {fold} 가지치기")
                tf.keras.backend.clear_session()
                raise optuna.exceptions.TrialPruned()

        # 메모리 정리
        tf.keras.backend.clear_session()

    if not all_fold_reports:
        logging.error("집계할 결과 없음")
        return None

    logging.info("===== 모든 Fold 완료. 최종 집계 =====")
    final_report = pd.concat(all_fold_reports)
    avg_macro_f1 = float(final_report.loc['macro avg', 'f1-score'].mean())
    std_macro_f1 = float(final_report.loc['macro avg', 'f1-score'].std())
    logging.info(f"LOSO CV Macro F1: {avg_macro_f1:.4f} ± {std_macro_f1:.4f}")

    final_report_path = os.path.join(cfg['paths']['OUTPUT_DIR'], 'models', 'final_loso_cv_report.csv')
    os.makedirs(os.path.dirname(final_report_path), exist_ok=True)
    final_report.to_csv(final_report_path)

    if cfg['settings']['SAVE_VISUALIZATIONS']:
        if all_fold_histories:
            aggregated_history_df = pd.concat(all_fold_histories, ignore_index=True)
            agg_history_plot_path = os.path.join(cfg['paths']['OUTPUT_DIR'], 'models', 'aggregated_training_history.png')
            utils.plot_aggregated_loso_history(aggregated_history_df, agg_history_plot_path)
            logging.info(f"종합 학습 그래프 저장: {agg_history_plot_path}")

        loso_plot_path = os.path.join(cfg['paths']['OUTPUT_DIR'], 'models', 'loso_cv_results_boxplot.png')
        utils.plot_loso_results(final_report, loso_plot_path)
        logging.info(f"LOSO 그래프 저장: {loso_plot_path}")

    logging.info(f"파이프라인 완료. 결과 폴더: '{cfg['paths']['OUTPUT_DIR']}'")
    return avg_macro_f1


if __name__ == '__main__':
    logging.warning("이 스크립트는 단독 실행용이 아닙니다. run.py를 통해 호출하세요.")
