"""
V2M_mirero (TensorFlow)
=======================
EfficientNetV2M backbone + GlobalAveragePooling + Dense(softmax).
"""
import tensorflow as tf


def build_V2M_mirero(num_classes: int, input_size: int = 480):
    inputs = tf.keras.layers.Input(shape=(input_size, input_size, 3))
    backbone = tf.keras.applications.EfficientNetV2M(
        include_top=False,
        weights=None,
        input_shape=(input_size, input_size, 3),
    )
    x = backbone(inputs)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs, name="V2M_mirero")
