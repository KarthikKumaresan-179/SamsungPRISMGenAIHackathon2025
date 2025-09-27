import tensorflow as tf
import pandas as pd
import os
import kagglehub


import tensorflow as tf
import os

AUTOTUNE = tf.data.AUTOTUNE

def read_image(path, img_size=(224,224), grayscale=True):
    img = tf.io.read_file(path)
    img = tf.io.decode_image(img, channels=1 if grayscale else 3, expand_animations=False)
    img = tf.image.convert_image_dtype(img, tf.float32)  # [0,1]
    img = tf.image.resize(img, img_size)
    return img

def basic_augment(image):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, 0.05)
    image = tf.image.random_contrast(image, 0.95, 1.05)
    return image

def make_dataset_from_directory(base_dir, img_size=(224,224), batch_size=16, grayscale=True):
    """
    Directly creates datasets from Coronahack Chest X-ray directory structure.
    Expects: base_dir/train/* , base_dir/test/*
    """
    train_dir = os.path.join(base_dir, "train")
    test_dir = os.path.join(base_dir, "test")

    # Train dataset
    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        image_size=img_size,
        batch_size=batch_size,
        color_mode="grayscale" if grayscale else "rgb"
    )

    # Test dataset
    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        image_size=img_size,
        batch_size=batch_size,
        color_mode="grayscale" if grayscale else "rgb"
    )

    # Split test into validation + test (50/50)
    val_size = int(0.5 * len(test_ds))
    val_ds = test_ds.take(val_size)
    test_ds = test_ds.skip(val_size)

    # Optimize
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
    test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

    return train_ds, val_ds, test_ds


# -------------------------------
# Example usage
# -------------------------------
if __name__ == "__main__":
    dataset_path = r"C:\Users\HP\Desktop\scan_cnn_tf\Coronahack-Chest-XRay-Dataset"  # <-- your local folder
    train_ds, val_ds, test_ds = make_dataset_from_directory(dataset_path)

    print("Classes:", train_ds.class_names)
    print("Train batches:", len(train_ds))
    print("Validation batches:", len(val_ds))
    print("Test batches:", len(test_ds))




'''AUTOTUNE = tf.data.AUTOTUNE  # updated (no need experimental in latest TF)

def fetch_kaggle_dataset(dataset_name="nih-chest-xrays/data"):
    """
    Fetch dataset from KaggleHub and return dataset path.
    """
    path = kagglehub.dataset_download(dataset_name)
    print("✅ Dataset fetched at:", path)
    return path

def read_image(path, img_size=(224,224), grayscale=True):
    img = tf.io.read_file(path)
    img = tf.io.decode_image(img, channels=1 if grayscale else 3, expand_animations=False)
    img = tf.image.convert_image_dtype(img, tf.float32)  # scale [0,1]
    img = tf.image.resize(img, img_size)
    return img

def basic_augment(image):
    # simple, safe medical augmentations
    image = tf.image.random_flip_left_right(image)
    # small random brightness/contrast changes
    image = tf.image.random_brightness(image, 0.05)
    image = tf.image.random_contrast(image, 0.95, 1.05)
    return image

def make_dataset(df, batch_size=16, img_size=(224,224), shuffle=True, augment=False, grayscale=True):
    """
    df must have columns: 'path' and 'label' (label numeric). For multi-label, label can be vector.
    """
    paths = df['path'].values
    labels = df['label'].values.astype('float32')

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    if shuffle:
        ds = ds.shuffle(buffer_size=len(paths), seed=42)

    def _load(path, label):
        img = read_image(path, img_size=img_size, grayscale=grayscale)
        if augment:
            img = basic_augment(img)
        return img, label

    ds = ds.map(_load, num_parallel_calls=AUTOTUNE)

    ds = ds.batch(batch_size).prefetch(AUTOTUNE)
    return ds


# -------------------------------
# Example usage
# -------------------------------
if __name__ == "__main__":
    # 1. Fetch dataset automatically
    dataset_path = fetch_kaggle_dataset("praveengovi/coronahack-chest-xraydataset")



    # 2. Assume dataset has "labels.csv" with 'path' and 'label' columns
    labels_csv = os.path.join(dataset_path, "labels.csv")
    if os.path.exists(labels_csv):
        df = pd.read_csv(labels_csv)
        train_df = df.sample(frac=0.8, random_state=42)  # simple split
        val_df = df.drop(train_df.index)

        train_ds = make_dataset(train_df, batch_size=16, augment=True)
        val_ds = make_dataset(val_df, batch_size=16, augment=False)

        print("Train/Val datasets ready:", train_ds, val_ds)
    else:
        print("⚠️ No labels.csv found in dataset. Please check dataset structure.")'''
