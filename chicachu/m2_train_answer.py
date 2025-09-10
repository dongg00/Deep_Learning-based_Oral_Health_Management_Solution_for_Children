# m2_train_answer.py
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.utils import class_weight
import logging
import utils

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
try:
    if tf.config.list_physical_devices('GPU'):
        from tensorflow.keras import mixed_precision
        mixed_precision.set_global_policy('mixed_float16')
        logging.info("Mixed Precision 활성화(GPU).")
    else:
        logging.info("GPU 없음: float32로 학습.")
except Exception as e:
    logging.warning(f"Mixed Precision 설정 불가: {e}")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
def _conv_frontend(x, cfg):
    """
    cfg['model_arch']['conv_frontend'] 설정을 읽어 1D Conv 프론트엔드를 붙입니다.
    SeparableConv1D는 kernel_regularizer를 받지 않으므로 전달 금지!
    """
    cf = cfg['model_arch'].get('conv_frontend', {})
    if not cf or not cf.get('enabled', False):
        return x

    filters  = int(cf.get('filters', 64))
    kernel   = int(cf.get('kernel_size', 5))
    strides  = int(cf.get('strides', 1))
    pool     = int(cf.get('pool_size', 2))
    dropout  = float(cf.get('dropout', 0.1))
    separable= bool(cf.get('separable', True))
    l2       = float(cf.get('l2', 1e-4))

    if separable:
        Conv = layers.SeparableConv1D
        conv_kwargs = dict(
            filters=filters, kernel_size=kernel, strides=strides,
            padding='same', activation='relu',
            depthwise_regularizer=regularizers.l2(l2),
            pointwise_regularizer=regularizers.l2(l2),
        )
    else:
        Conv = layers.Conv1D
        conv_kwargs = dict(
            filters=filters, kernel_size=kernel, strides=strides,
            padding='same', activation='relu',
            kernel_regularizer=regularizers.l2(l2),
        )

    x = Conv(**conv_kwargs)(x)
    x = layers.MaxPool1D(pool_size=pool)(x)
    if cf.get('use_ln', True):
        x = layers.LayerNormalization()(x)
    if dropout > 0:
        x = layers.Dropout(dropout)(x)
    return x


# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
def build_model(input_shape, n_classes, cfg):
    """
    - variant: 'gru' | 'bigru' | 'gru_attn' | 'gru_sa' | 'cnn_gru'
    - gru1_units, gru2_units, dense_units, dropout_rate, recurrent_dropout, use_layer_norm
    - conv_frontend: {enabled, filters, kernel_size, strides, pool_size, separable, dropout, l2, use_ln}
    - self_attention: {num_heads, key_dim, dropout}  # variant='gru_sa'일 때 사용
    """
    p = cfg.get('model_arch', {})
    variant = p.get('variant', 'gru_attn')
    rdrop = float(p.get('recurrent_dropout', 0.0))
    use_ln = bool(p.get('use_layer_norm', False))
    l2r = float(p.get('l2', 1e-4))

    inp = layers.Input(shape=input_shape)
    # 혼합정밀 사용해도 안전하도록 입력을 명시적 float32로 캐스팅
    x = layers.Rescaling(1.0, dtype='float32', name="cast_to_float32")(inp)
    x = layers.GaussianNoise(0.01)(x)

    if variant == 'cnn_gru' or p.get('conv_frontend', {}).get('enabled', False):
        x = _conv_frontend(x, cfg)

    gru1 = int(p.get('gru1_units', 128))
    gru2 = int(p.get('gru2_units', 64))
    dropout_gru = 0.3

    def _maybe_ln(t):
        return layers.LayerNormalization()(t) if use_ln else t

    if variant == 'bigru':
        x = layers.Bidirectional(
            layers.GRU(units=gru1, return_sequences=True,
                       dropout=dropout_gru, recurrent_dropout=rdrop,
                       kernel_regularizer=regularizers.l2(l2r))
        )(x)
        x = _maybe_ln(x)
        if gru2 > 0:
            x = layers.Bidirectional(
                layers.GRU(units=gru2, return_sequences=True,
                           dropout=dropout_gru, recurrent_dropout=rdrop,
                           kernel_regularizer=regularizers.l2(l2r))
            )(x)
            x = _maybe_ln(x)
    else:
        x = layers.GRU(units=gru1, return_sequences=True,
                       dropout=dropout_gru, recurrent_dropout=rdrop,
                       kernel_regularizer=regularizers.l2(l2r))(x)
        x = _maybe_ln(x)
        if gru2 > 0:
            x = layers.GRU(units=gru2, return_sequences=True,
                           dropout=dropout_gru, recurrent_dropout=rdrop,
                           kernel_regularizer=regularizers.l2(l2r))(x)
            x = _maybe_ln(x)

    if variant == 'gru_attn':
        x = layers.Attention()([x, x])
        x = _maybe_ln(x)
    elif variant == 'gru_sa':
        sa = p.get('self_attention', {})
        num_heads = int(sa.get('num_heads', 4))
        key_dim = int(sa.get('key_dim', max(16, gru2 if gru2 > 0 else max(16, gru1 // max(1, num_heads)))))
        sa_dropout = float(sa.get('dropout', 0.1))
        x = layers.MultiHeadAttention(num_heads=num_heads, key_dim=key_dim,
                                      dropout=sa_dropout)(x, x)
        x = layers.Add()([x, layers.LayerNormalization()(x)])
        x = _maybe_ln(x)

    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(units=int(p.get('dense_units', 64)), activation='relu',
                     kernel_regularizer=regularizers.l2(l2r))(x)
    x = layers.Dropout(float(p.get('dropout_rate', 0.5)))(x)
    out = layers.Dense(n_classes, activation='softmax', dtype=tf.float32)(x)

    model = models.Model(inp, out)
    opt = tf.keras.optimizers.Adam(learning_rate=cfg['training']['learning_rate'])
    loss = tf.keras.losses.SparseCategoricalCrossentropy()
    model.compile(optimizer=opt, loss=loss, metrics=['accuracy'])

    logging.info(f"[ARCH] variant={variant} | gru1={gru1}, gru2={gru2}, dense={p.get('dense_units', 64)}"
                 f" | LN={use_ln} | rdrop={rdrop} | conv_frontend={p.get('conv_frontend', {}).get('enabled', False)}")
    if variant == 'gru_sa':
        logging.info(f"[ARCH] self_attention={p.get('self_attention', {})}")

    return model


# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
def main(X_train, y_train, X_val, y_val, n_classes, model_save_path, cfg):
    logging.info(f"===== 모델 학습 시작 (저장 경로: {model_save_path}) =====")
    # 재현성(옵션)
    try:
        tf.keras.utils.set_random_seed(int(cfg['settings'].get('SEED', 42)))
    except Exception:
        pass

    model = build_model(X_train.shape[1:], n_classes, cfg)
    model.summary(print_fn=logging.info)

    # ✅ 검증 세트 유무
    has_val = (X_val is not None) and (y_val is not None) and (len(y_val) > 0)

    # 콜백 구성
    if has_val:
        es = EarlyStopping(monitor='val_loss',
                           patience=int(cfg['training']['early_stopping_patience']),
                           verbose=1, restore_best_weights=True)
        ckpt_monitor = 'val_loss'
        val_data = (X_val, y_val)
    else:
        es = EarlyStopping(monitor='loss',
                           patience=min(5, int(cfg['training']['early_stopping_patience'])),
                           verbose=1, restore_best_weights=True)
        ckpt_monitor = 'loss'
        val_data = None
        logging.info("검증 세트 없음: EarlyStopping/Checkpoint는 'loss'로 모니터링합니다.")

    rlrop = ReduceLROnPlateau(monitor=ckpt_monitor, factor=0.5, patience=10,
                              min_lr=1e-6, verbose=1)

    # 새 Keras 포맷으로 저장
    save_path = os.path.splitext(model_save_path)[0] + '.keras'
    ckpt = ModelCheckpoint(filepath=save_path, monitor=ckpt_monitor,
                           save_best_only=True, verbose=1)

    # 클래스 가중치
    classes = np.unique(y_train)
    weights = class_weight.compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
    class_weights = dict(zip(classes.tolist(), weights.tolist()))
    logging.info(f"클래스 가중치: {class_weights}")

    history = model.fit(
        X_train, y_train,
        validation_data=val_data,   # None이면 검증 완전 생략
        epochs=int(cfg['training']['epochs']),
        batch_size=int(cfg['training']['batch_size']),
        callbacks=[es, rlrop, ckpt],
        class_weight=class_weights,
        verbose=2
    )

    # 기록/그래프 저장 (검증 없을 때도 history는 존재)
    fold_output_dir = os.path.dirname(model_save_path)
    try:
        history_df = pd.DataFrame(history.history)
        history_csv_path = os.path.join(fold_output_dir, 'history.csv')
        history_df.to_csv(history_csv_path, index_label='epoch')
        logging.info(f"학습 기록 저장: {history_csv_path}")

        if cfg['settings'].get('SAVE_VISUALIZATIONS', False):
            history_plot_path = os.path.join(fold_output_dir, 'training_history.png')
            utils.plot_training_history(history, history_plot_path)
            logging.info(f"개별 학습 그래프 저장: {history_plot_path}")
    except Exception as e:
        logging.warning(f"학습 기록 저장/시각화 중 경고: {e}")

    logging.info("===== 모델 학습 완료 =====")
