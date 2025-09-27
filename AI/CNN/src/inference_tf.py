# src/inference_tf.py
import tensorflow as tf
import numpy as np
from models.parallel_cnn_tf import build_parallel_cnn
from tensorflow.keras.models import load_model
from PIL import Image

def preprocess_image(image_path, img_size=(224,224), grayscale=True):
    img = Image.open(image_path)
    if grayscale:
        img = img.convert('L')
    else:
        img = img.convert('RGB')
    img = img.resize(img_size)
    arr = np.array(img).astype('float32') / 255.0
    if grayscale:
        arr = np.expand_dims(arr, axis=-1)
    arr = np.expand_dims(arr, axis=0)  # batch dim
    return arr

def load_trained_model(checkpoint_path, input_shape=(224,224,1), num_classes=1):
    # If checkpoint saved as h5 from ModelCheckpoint, load directly
    try:
        model = tf.keras.models.load_model(checkpoint_path, compile=False)
        return model
    except Exception as e:
        # fallback: rebuild architecture and load weights
        model = build_parallel_cnn(input_shape=input_shape, num_classes=(num_classes if num_classes>1 else 1))
        model.load_weights(checkpoint_path)
        return model

def predict_image(model, image_path, threshold=0.5, class_names=None, img_size=(224,224), grayscale=True):
    arr = preprocess_image(image_path, img_size=img_size, grayscale=grayscale)
    preds = model.predict(arr)
    if preds.shape[-1] == 1:
        prob = float(preds[0,0])
        label = 1 if prob >= threshold else 0
        name = class_names[label] if class_names else str(label)
        return {'label': label, 'class': name, 'probability': prob}
    else:
        # multi-class softmax
        prob_vec = preds[0]
        idx = int(np.argmax(prob_vec))
        prob = float(prob_vec[idx])
        name = class_names[idx] if class_names else str(idx)
        return {'label': idx, 'class': name, 'probability': prob}
