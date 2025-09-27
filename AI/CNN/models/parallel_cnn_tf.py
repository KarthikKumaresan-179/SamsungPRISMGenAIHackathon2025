# models/parallel_cnn_tf.py
import tensorflow as tf
from tensorflow.keras import layers, Model

def ParallelConvBlock(inputs, out_filters, reduction=4, name=None):
    """
    Inception-style parallel block: 1x1, 3x3, factorized 5x5 (3x3x2), and pooling branch.
    inputs: Keras tensor
    out_filters: total filters after fuse
    """
    in_ch = inputs.shape[-1]
    mid = max(8, out_filters // reduction)

    # branch 1: 1x1
    b1 = layers.Conv2D(mid, 1, padding='same', use_bias=False)(inputs)
    b1 = layers.BatchNormalization()(b1)
    b1 = layers.ReLU()(b1)

    # branch 2: 3x3
    b2 = layers.Conv2D(mid, 3, padding='same', use_bias=False)(inputs)
    b2 = layers.BatchNormalization()(b2)
    b2 = layers.ReLU()(b2)

    # branch 3: factorized 5x5 => two 3x3
    b3 = layers.Conv2D(mid, 3, padding='same', use_bias=False)(inputs)
    b3 = layers.BatchNormalization()(b3)
    b3 = layers.ReLU()(b3)
    b3 = layers.Conv2D(mid, 3, padding='same', use_bias=False)(b3)
    b3 = layers.BatchNormalization()(b3)
    b3 = layers.ReLU()(b3)

    # branch 4: pooling
    b4 = layers.AveragePooling2D(pool_size=3, strides=1, padding='same')(inputs)
    b4 = layers.Conv2D(mid, 1, padding='same', use_bias=False)(b4)
    b4 = layers.BatchNormalization()(b4)
    b4 = layers.ReLU()(b4)

    # concat & fuse
    cat = layers.Concatenate(axis=-1)([b1, b2, b3, b4])
    fused = layers.Conv2D(out_filters, 1, padding='same', use_bias=False)(cat)
    fused = layers.BatchNormalization()(fused)

    # shortcut (1x1) if channel mismatch
    if in_ch != out_filters:
        shortcut = layers.Conv2D(out_filters, 1, padding='same', use_bias=False)(inputs)
        shortcut = layers.BatchNormalization()(shortcut)
    else:
        shortcut = inputs

    out = layers.Add()([fused, shortcut])
    out = layers.ReLU()(out)
    return out

def build_parallel_cnn(input_shape=(224,224,1), num_classes=2, dropout=0.4):
    inp = layers.Input(shape=input_shape)

    # stem
    x = layers.Conv2D(32, 3, padding='same', use_bias=False)(inp)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # blocks with downsampling
    x = ParallelConvBlock(x, 64)
    x = layers.MaxPool2D(pool_size=2)(x)

    x = ParallelConvBlock(x, 128)
    x = layers.MaxPool2D(pool_size=2)(x)

    x = ParallelConvBlock(x, 256)
    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(dropout)(x)

    if num_classes == 1:
        out = layers.Dense(1, activation='sigmoid')(x)   # binary (sigmoid)
    else:
        out = layers.Dense(num_classes, activation='softmax')(x)  # multi-class

    model = Model(inputs=inp, outputs=out)
    return model

if __name__ == "__main__":
    m = build_parallel_cnn((224,224,1), num_classes=2)
    m.summary()
