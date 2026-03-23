import multiprocessing

import numpy as np
import tensorflow as tf

def _parse_function(example_proto):
    # Define the specific keys found in the shards
    band_keys = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B11', 'B12']

    feature_description = {
        'classification': tf.io.FixedLenFeature([1], tf.float32),
    }

    for key in band_keys:
        feature_description[key] = tf.io.FixedLenFeature([129 * 129], tf.float32)

    parsed_features = tf.io.parse_single_example(example_proto, feature_description)

    # Reconstruct the 12-channel tensor
    bands = []
    for key in band_keys:
        band = tf.reshape(parsed_features[key], [129, 129, 1])
        bands.append(band)

    # Stack along the last axis to get (129, 129, 12)
    full_tensor = tf.concat(bands, axis=-1)

    # Get the label
    label = tf.cast(parsed_features['classification'][0], tf.int64)

    return full_tensor, label


def load_shards(filenames, batch_size=512, is_training=True, is_svm=False):
    cores = multiprocessing.cpu_count()
    dataset = tf.data.Dataset.from_tensor_slices(filenames)

    dataset = dataset.interleave(
        lambda x: tf.data.TFRecordDataset(x, compression_type='GZIP'),
        cycle_length=cores,
        block_length=16,
        num_parallel_calls=cores,
        deterministic=False
    )

    dataset = dataset.map(_parse_function, num_parallel_calls=cores)

    # Statistical reduction for svms
    if is_svm:
        dataset = dataset.map(calculate_spatial_stats, num_parallel_calls=cores)

    # Shuffle for training data
    if is_training:
        dataset = dataset.shuffle(buffer_size=1000)

    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    return dataset

def calculate_spatial_stats(image, label):
    flat_pixels = tf.reshape(image, [-1, 12])

    sorted_pixels = tf.sort(flat_pixels, axis=0)
    num_pixels = tf.cast(tf.shape(sorted_pixels)[0], tf.float32)

    idx_q1 = tf.cast(num_pixels * 0.25, tf.int32)
    idx_med = tf.cast(num_pixels * 0.5, tf.int32)
    idx_q3 = tf.cast(num_pixels * 0.75, tf.int32)

    mean = tf.reduce_mean(flat_pixels, axis=0)
    q1 = tf.gather(sorted_pixels, idx_q1)
    median = tf.gather(sorted_pixels, idx_med)
    q3 = tf.gather(sorted_pixels, idx_q3)
    maximum = tf.reduce_max(flat_pixels, axis=0)

    # Stack stats into (5, 12) vector and flatten to (60)
    stats_vector = tf.stack([mean, q1, median, q3, maximum], axis=0)
    flat_features = tf.reshape(stats_vector, [-1])

    return flat_features, label

def get_svm_data(dataset):
    """
    Converts the TF dataset directly into NumPy arrays for scikit-learn.
    """
    X_list = []
    y_list = []

    for features, labels in dataset:
        X_list.append(features.numpy())
        y_list.append(labels.numpy())

    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)

    return X, y
