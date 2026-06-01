"""
B3_mirero (TensorFlow)
======================
EfficientNetB3 backbone + GlobalAveragePooling + Dense(softmax).

사용자 코드 패턴 매칭:
    backbone = EfficientNetB3(include_top=False, input_shape=(input_size,input_size,3))
    x = GlobalAveragePooling2D()(backbone.output)
    out = Dense(num_classes, activation="softmax")(x)
    model = Model(backbone.input, out)
"""
import tensorflow as tf


def build_B3_mirero(num_classes: int, input_size: int = 300):
    inputs = tf.keras.layers.Input(shape=(input_size, input_size, 3))
    backbone = tf.keras.applications.EfficientNetB3(
        include_top=False,
        weights=None,
        input_shape=(input_size, input_size, 3),
    )
    x = backbone(inputs)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs, name="B3_mirero")
