# m3_evaluate_answer.py
import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import logging
import utils

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def _load_model_robust(model_path: str):
    """Keras 모델을 최대한 안전하게 로드합니다(.keras 우선, 레거시 safe_mode=False 폴백)."""
    try:
        return tf.keras.models.load_model(model_path)
    except Exception as e1:
        # 레거시 아티팩트(Lambda 포함 등) 대비
        try:
            logging.warning("일반 로드 실패 → safe_mode=False로 재시도합니다(레거시 모델).")
            return tf.keras.models.load_model(model_path, safe_mode=False)
        except Exception:
            logging.error(f"모델 로드 실패: {model_path}", exc_info=True)
            return None


def main(
    X_test,
    y_test,
    class_names,
    model_path,
    output_dir,
    cfg,
    meta_keys=None  # ✅ 선택 인자: 제공되면 원시 예측과 함께 저장
):
    """학습된 모델을 평가하고 리포트/혼동행렬/원시예측을 저장합니다."""
    logging.info(f"===== 모델 평가 시작 (모델: {model_path}) =====")

    # 새 포맷(.keras) 우선
    base, ext = os.path.splitext(model_path)
    candidate_paths = []
    if os.path.exists(base + '.keras'):
        candidate_paths.append(base + '.keras')
    if os.path.exists(model_path):
        candidate_paths.append(model_path)
    # 중복 제거 후 순서 유지
    candidate_paths = list(dict.fromkeys(candidate_paths))
    if not candidate_paths:
        logging.error(f"모델 파일을 찾을 수 없습니다: {model_path}")
        return None

    model = None
    for mpth in candidate_paths:
        model = _load_model_robust(mpth)
        if model is not None:
            model_path = mpth
            break
    if model is None:
        return None

    # 예측
    y_pred_proba = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_proba, axis=1)

    # 리포트(모든 클래스 고정 집합으로 리포트 생성)
    n_classes = len(class_names)
    labels_fixed = np.arange(n_classes)
    report = classification_report(
        y_test,
        y_pred,
        labels=labels_fixed,
        target_names=list(class_names),
        output_dict=True,
        zero_division=0
    )
    report_df = pd.DataFrame(report).transpose()
    logging.info("\n" + report_df.to_string())

    # 저장 디렉토리
    os.makedirs(output_dir, exist_ok=True)

    # 리포트 저장
    report_path = os.path.join(output_dir, 'classification_report.csv')
    report_df.to_csv(report_path)
    logging.info(f"분류 리포트 저장: {report_path}")

    # 혼동행렬
    cm = confusion_matrix(y_test, y_pred, labels=labels_fixed)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted'); plt.ylabel('True')
    cm_path = os.path.join(output_dir, 'confusion_matrix.png')
    plt.savefig(cm_path); plt.close()
    logging.info(f"혼동 행렬 저장: {cm_path}")

    # 원시 예측 저장(+ 옵션 meta_keys)
    if cfg['settings'].get('EVAL', {}).get('save_raw_predictions', True):
        pred_path = os.path.join(output_dir, 'raw_predictions.npz')
        if meta_keys is not None and len(meta_keys) == len(y_test):
            np.savez_compressed(
                pred_path,
                y_true=y_test,
                y_pred=y_pred,
                y_proba=y_pred_proba,
                meta_keys=np.asarray(meta_keys)
            )
        else:
            np.savez_compressed(
                pred_path,
                y_true=y_test,
                y_pred=y_pred,
                y_proba=y_pred_proba
            )
        logging.info(f"원시 예측 저장: {pred_path}")

    # 클래스별 성능 그래프
    if cfg['settings'].get('SAVE_VISUALIZATIONS', False):
        metrics_plot_path = os.path.join(output_dir, 'per_class_metrics.png')
        utils.plot_per_class_metrics(report_df, metrics_plot_path)
        logging.info(f"클래스별 성능 그래프 저장: {metrics_plot_path}")

    logging.info("===== 모델 평가 완료 =====")
    return report_df


if __name__ == "__main__":
    logging.warning("이 스크립트는 단독 실행용이 아닙니다. run.py → main.py에서 호출하세요.")
