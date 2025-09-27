# models/losses.py
import tensorflow as tf

def binary_focal_loss(gamma=2.0, alpha=0.25):
    def loss_fn(y_true, y_pred):
        # y_pred: probability (sigmoid) for binary case, or logits if using from_logits
        y_true = tf.cast(y_true, tf.float32)
        # clip
        eps = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, eps, 1.0 - eps)
        p_t = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
        alpha_t = tf.where(tf.equal(y_true, 1), alpha, 1 - alpha)
        loss = -alpha_t * ((1 - p_t) ** gamma) * tf.math.log(p_t)
        return tf.reduce_mean(loss)
    return loss_fn
