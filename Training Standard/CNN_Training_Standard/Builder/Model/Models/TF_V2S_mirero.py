"""
V2S_mirero (TensorFlow)
=======================
EfficientNetV2S backbone + GlobalAveragePooling + Dense(softmax).

사용자 코드 그대로:
    backbone = EfficientNetV2S(include_top=False, input_shape=(image_size,image_size,3))
    x = GlobalAveragePooling2D()(backbone.output)
    out = Dense(classes, activation="softmax")(x)
    model = Model(backbone.input, out)
"""
import tensorflow as tf


def build_V2S_mirero(num_classes: int, input_size: int = 384):
    inputs = tf.keras.layers.Input(shape=(input_size, input_size, 3))
    backbone = tf.keras.applications.EfficientNetV2S(
        include_top=False,
        weights=None,
        input_shape=(input_size, input_size, 3),
    )
    x = backbone(inputs)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs, name="V2S_mirero")
