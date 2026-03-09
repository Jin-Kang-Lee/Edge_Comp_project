import tensorflow as tf

from .constants import FEATURE_DIM


def build_encoder(input_dim: int = FEATURE_DIM) -> tf.keras.Model:
    model = tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer(input_shape=(input_dim,)),
            tf.keras.layers.Dense(8, activation="relu"),
            tf.keras.layers.Dense(4, activation=None),
        ],
        name="svdd_encoder",
    )
    return model
