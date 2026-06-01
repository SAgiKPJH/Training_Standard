"""
B7_mirero (TensorFlow)
======================
EfficientNetB7 backbone + GlobalAveragePooling + Dense(softmax).
"""
import tensorflow as tf


def build_B7_mirero(num_classes: int, input_size: int = 600):
    inputs = tf.keras.layers.Input(shape=(input_size, input_size, 3))
    backbone = tf.keras.applications.EfficientNetB7(
        include_top=False,
        weights=None,
        input_shape=(input_size, input_size, 3),
    )
    x = backbone(inputs)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs, name="B7_mirero")
