import numpy as np
import tensorflow as tf


def compute_center(model: tf.keras.Model, batch: np.ndarray) -> np.ndarray:
    embeddings = model(batch, training=False).numpy()
    center = np.mean(embeddings, axis=0)
    return center.astype(np.float32)


def svdd_loss(center: tf.Tensor):
    def _loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        del y_true
        diff = y_pred - center
        dist_sq = tf.reduce_sum(tf.square(diff), axis=1)
        return tf.reduce_mean(dist_sq)

    return _loss


def compute_distances(
    model: tf.keras.Model, data: np.ndarray, center: np.ndarray, batch_size: int
) -> np.ndarray:
    ds = tf.data.Dataset.from_tensor_slices(data).batch(batch_size)
    dists = []
    center_tf = tf.constant(center, dtype=tf.float32)
    for batch in ds:
        emb = model(batch, training=False)
        diff = emb - center_tf
        dist_sq = tf.reduce_sum(tf.square(diff), axis=1)
        dists.append(dist_sq.numpy())
    return np.concatenate(dists, axis=0)
