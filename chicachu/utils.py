import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# 스타일(선택): 보기 편하게 whitegrid
sns.set_theme(style="whitegrid")

try:
    import koreanize_matplotlib  # noqa: F401
except ImportError:
    print("koreanize-matplotlib 라이브러리가 설치되지 않았습니다. 한글이 깨질 수 있습니다.")


def _ensure_dir(path: str):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)


def plot_class_distribution(y, class_names, output_path):
    plt.figure(figsize=(12, 8))
    class_counts = pd.Series(y).value_counts().sort_index()

    # 인덱스가 정수라면 class_names 매핑, 아니라면 문자열 그대로 사용
    try:
        x_labels = [class_names[i] for i in class_counts.index]
    except Exception:
        x_labels = class_counts.index.astype(str).tolist()

    sns.barplot(x=x_labels, y=class_counts.values)
    plt.title('Class Distribution of Samples')
    plt.xlabel('Class Name')
    plt.ylabel('Number of Samples (Windows)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    _ensure_dir(output_path)
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_sample_landmarks(sample, output_path):
    first_frame = sample[0]
    num_hand_lms = 21
    if len(first_frame) < num_hand_lms * 2:
        return
    hand_lms = first_frame[:num_hand_lms * 2].reshape(num_hand_lms, 2)
    face_lms = first_frame[num_hand_lms * 2:].reshape(-1, 2)

    plt.figure(figsize=(8, 8))
    plt.scatter(hand_lms[:, 0], hand_lms[:, 1], c='blue', label='Hand Landmarks', s=10)
    if face_lms.size > 0:
        plt.scatter(face_lms[:, 0], face_lms[:, 1], c='red', label='Face Landmarks', s=10)
    plt.title('Visualized Sample (1st Frame after Normalization)')
    plt.xlabel('X (normalized)'); plt.ylabel('Y (normalized)')
    plt.grid(True); plt.axis('equal'); plt.legend()
    _ensure_dir(output_path)
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_training_history(history, output_path):
    history_df = pd.DataFrame(history.history)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    # Loss
    cols_loss = [c for c in ['loss', 'val_loss'] if c in history_df.columns]
    if cols_loss:
        history_df[cols_loss].plot(ax=ax1)
    ax1.set_title('Model Loss'); ax1.set_xlabel('Epochs'); ax1.set_ylabel('Loss'); ax1.grid(True)

    # Accuracy
    cols_acc = [c for c in ['accuracy', 'val_accuracy'] if c in history_df.columns]
    if cols_acc:
        history_df[cols_acc].plot(ax=ax2)
        ax2.legend()
    else:
        if 'accuracy' in history_df.columns:
            ax2.plot(history_df.index, history_df['accuracy'], label='accuracy')
        if 'val_accuracy' in history_df.columns:
            ax2.plot(history_df.index, history_df['val_accuracy'], label='val_accuracy')
        if ax2.has_data():
            ax2.legend()

    ax2.set_title('Model Accuracy'); ax2.set_xlabel('Epochs'); ax2.set_ylabel('Accuracy'); ax2.grid(True)

    plt.tight_layout()
    _ensure_dir(output_path)
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_per_class_metrics(report_df, output_path):
    # 클래스별 행만 남기고 평균행 제거
    plot_df = report_df.drop([x for x in ['accuracy', 'macro avg', 'weighted avg'] if x in report_df.index],
                             errors='ignore')
    cols = [c for c in ['f1-score', 'precision', 'recall'] if c in report_df.columns]
    if plot_df.empty or not cols:
        return

    plt.figure(figsize=(15, 8))
    ax = plt.gca()
    plot_df[cols].plot(kind='bar', ax=ax)
    plt.title('Per-Class Performance Metrics')
    plt.ylabel('Score')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y')
    plt.tight_layout()
    _ensure_dir(output_path)
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_loso_results(report_df, output_path):
    """
    final_loso_cv_report.csv 형태(여러 fold의 classification_report가 concat된 DF)에서
    fold별 macro F1을 안정적으로 뽑아 박스/분포를 그림.
    """
    if 'f1-score' not in report_df.columns:
        return

    # 1) fold 컬럼이 있을 때: fold별 macro avg만 추출
    macro_f1_scores = None
    if 'fold' in report_df.columns:
        tmp = report_df.reset_index().rename(columns={'index': 'label'})
        tmp = tmp[tmp['label'] == 'macro avg']
        if not tmp.empty:
            # fold -> f1-score 매핑
            macro_f1_scores = tmp.set_index('fold')['f1-score'].sort_index()

    # 2) fold 컬럼이 없고, 단일 러닝일 때 대비
    if macro_f1_scores is None:
        try:
            locv = report_df.loc['macro avg']['f1-score']
            if isinstance(locv, pd.DataFrame):
                macro_f1_scores = locv.squeeze()
            else:
                macro_f1_scores = pd.Series([float(locv)], index=[1])
        except Exception:
            return

    # 시각화 (가로 박스플롯 → x= 사용)
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=macro_f1_scores)                 # ✅ data 대신 x
    sns.stripplot(x=macro_f1_scores, color=".25")  # ✅ 가로 stripplot
    plt.title('LOSO Cross-Validation Results (Macro F1-Score per Fold)')
    plt.xlabel(f'Macro F1-Score (avg {macro_f1_scores.mean():.4f} ± {macro_f1_scores.std():.4f})')
    plt.ylabel('')  # 축 라벨 비우기
    plt.grid(True, axis='x')
    _ensure_dir(output_path)
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_aggregated_loso_history(all_histories_df, output_path):
    if 'epoch' not in all_histories_df.columns:
        return
    agg_df = all_histories_df.groupby('epoch').agg(['mean', 'std'])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))

    # Loss
    for key, label in [('loss', 'Training Loss'), ('val_loss', 'Validation Loss')]:
        if (key, 'mean') in agg_df.columns:
            m = agg_df[(key, 'mean')]
            ax1.plot(agg_df.index, m, label=f'{label} (Mean)')
            if (key, 'std') in agg_df.columns:
                s = agg_df[(key, 'std')].fillna(0)
                ax1.fill_between(agg_df.index, m - s, m + s, alpha=0.2)
    ax1.set_title('Aggregated Model Loss across All Folds')
    ax1.set_ylabel('Loss'); ax1.grid(True); ax1.legend()

    # Accuracy
    for key, label in [('accuracy', 'Training Accuracy'), ('val_accuracy', 'Validation Accuracy')]:
        if (key, 'mean') in agg_df.columns:
            m = agg_df[(key, 'mean')]
            ax2.plot(agg_df.index, m, label=f'{label} (Mean)')
            if (key, 'std') in agg_df.columns:
                s = agg_df[(key, 'std')].fillna(0)
                ax2.fill_between(agg_df.index, m - s, m + s, alpha=0.2)
    ax2.set_title('Aggregated Model Accuracy across All Folds')
    ax2.set_xlabel('Epochs'); ax2.set_ylabel('Accuracy'); ax2.grid(True); ax2.legend()

    plt.tight_layout()
    _ensure_dir(output_path)
    plt.savefig(output_path, dpi=150)
    plt.close()
