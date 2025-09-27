# src/train_tf.py
import os
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from models.parallel_cnn_tf import build_parallel_cnn
from models.losses import binary_focal_loss
from src.dataset_tf import make_dataset
import sklearn
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

def compile_and_train(df, model_save_dir='checkpoints', img_size=(224,224), batch_size=16,
                      epochs=20, lr=1e-4, num_classes=1, use_focal=False):

    os.makedirs(model_save_dir, exist_ok=True)

    # split (patient-level split recommended if you have patient_id)
    df_train, df_val = train_test_split(df, test_size=0.15, random_state=42, stratify=df['label'])

    train_ds = make_dataset(df_train, batch_size=batch_size, img_size=img_size,
                            shuffle=True, augment=True, grayscale=True)
    val_ds = make_dataset(df_val, batch_size=batch_size*2, img_size=img_size,
                          shuffle=False, augment=False, grayscale=True)

    model = build_parallel_cnn(input_shape=(img_size[0], img_size[1], 1), num_classes=num_classes)
    optimizer = Adam(learning_rate=lr)

    # choose loss
    if use_focal and num_classes == 1:
        loss = binary_focal_loss()
    else:
        if num_classes == 1:
            loss = 'binary_crossentropy'
        else:
            loss = 'sparse_categorical_crossentropy'

    # metrics: AUC (ROC)
    metrics = [tf.keras.metrics.AUC(name='auc'), tf.keras.metrics.BinaryAccuracy(name='accuracy')]

    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    # callbacks
    checkpoint = ModelCheckpoint(os.path.join(model_save_dir, 'best.h5'), monitor='val_auc',
                                 mode='max', save_best_only=True, verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_auc', mode='max', patience=3, factor=0.5, verbose=1)
    early = EarlyStopping(monitor='val_auc', mode='max', patience=8, restore_best_weights=True, verbose=1)

    # class weights (if severe imbalance)
    # compute class weights
    classes = df['label'].values
    unique = np.unique(classes)
    if len(unique) == 2:
        from sklearn.utils import class_weight
        cw = class_weight.compute_class_weight('balanced', classes=unique, y=classes)
        class_weight_dict = {int(unique[i]): float(cw[i]) for i in range(len(unique))}
    else:
        class_weight_dict = None

    history = model.fit(train_ds,
                        validation_data=val_ds,
                        epochs=epochs,
                        callbacks=[checkpoint, reduce_lr, early],
                        class_weight=class_weight_dict)

    return model, history
