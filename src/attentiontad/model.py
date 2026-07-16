"""AttentionTAD model definition.

TensorFlow is imported lazily so preprocessing can be used without installing
the optional training dependencies.
"""

from __future__ import annotations

from typing import Sequence


def _load_tensorflow():
    try:
        import tensorflow as tf
    except ImportError as exc:  # pragma: no cover - depends on optional package
        raise RuntimeError(
            "TensorFlow is required for model training. "
            "Install the training dependencies with: pip install -e '.[train]'"
        ) from exc
    return tf


def create_attentiontad_model(
    input_shape: Sequence[int] = (10_000, 4),
    num_filters: int = 256,
    kernel_size: int = 19,
    dense_units: int = 64,
):
    """Build the attention-guided convolutional classifier.

    Parameters
    ----------
    input_shape:
        Sequence length and one-hot channel count.
    num_filters:
        Number of convolutional filters and feature-weighting units.
    kernel_size:
        Width of the one-dimensional convolutional kernels.
    dense_units:
        Width of the final hidden layer.
    """

    tf = _load_tensorflow()
    keras = tf.keras
    layers = keras.layers

    inputs = keras.Input(shape=tuple(input_shape), name="sequence")
    features = layers.Conv1D(
        filters=num_filters,
        kernel_size=kernel_size,
        activation="relu",
        padding="valid",
        name="sequence_convolution",
    )(inputs)
    features = layers.GlobalMaxPooling1D(name="global_max_pooling")(features)

    weights = layers.Dense(
        num_filters,
        activation="tanh",
        name="feature_weighting_hidden",
    )(features)
    weights = layers.Dense(
        num_filters,
        activation="softmax",
        name="feature_weights",
    )(weights)
    weighted_features = layers.Multiply(name="weighted_features")(
        [features, weights]
    )

    hidden = layers.Dense(dense_units, activation="relu", name="classifier_hidden")(
        weighted_features
    )
    outputs = layers.Dense(1, activation="sigmoid", name="boundary_probability")(
        hidden
    )
    return keras.Model(inputs=inputs, outputs=outputs, name="AttentionTAD")
