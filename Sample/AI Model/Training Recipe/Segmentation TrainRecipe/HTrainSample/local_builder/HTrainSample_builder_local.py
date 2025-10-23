import os
import numpy as np
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.applications.efficientnet import EfficientNetB3
from keras.layers import Conv2D, BatchNormalization, Activation, Add, UpSampling2D, concatenate
import matplotlib.pyplot as plt
import os
import tensorflow as tf

def CBAR_block(input_tensor, num_filters):
    x = Conv2D(filters=num_filters, kernel_size=3, padding='same')(input_tensor)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters=num_filters, kernel_size=3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    xd = Conv2D(filters=num_filters, kernel_size=1)(input_tensor)
    x = Add()([x, xd])
    return x

def get_efficient(input_shape):
    return EfficientNetB3(
        include_top=False,
        weights='imagenet',
        input_tensor=None,
        input_shape=input_shape,
        pooling=None
    )

def efficient_unet(input_shape=(224, 224, 3), num_classes=3):
    encoder_model = get_efficient(input_shape=input_shape)
    new_input = encoder_model.input
    encoder_output = encoder_model.get_layer(name='block7a_project_bn').output

    fn_bottle_neck = encoder_output.shape[-1]
    bottleneck = CBAR_block(encoder_output, fn_bottle_neck)

    c1 = encoder_model.get_layer(name='block5c_drop').output
    fn_1 = c1.shape[-1]
    up1 = UpSampling2D()(bottleneck)
    cat1 = concatenate([up1, c1], axis=3)
    dec1 = CBAR_block(cat1, fn_1)

    c2 = encoder_model.get_layer(name='block3b_drop').output
    fn_2 = c2.shape[-1]
    up2 = UpSampling2D()(dec1)
    cat2 = concatenate([up2, c2], axis=3)
    dec2 = CBAR_block(cat2, fn_2)

    c3 = encoder_model.get_layer(name='block2b_drop').output
    fn_3 = c3.shape[-1]
    up3 = UpSampling2D()(dec2)
    cat3 = concatenate([up3, c3], axis=3)
    dec3 = CBAR_block(cat3, fn_3)

    c4 = encoder_model.get_layer(name='block1a_project_bn').output
    fn_4 = c4.shape[-1]
    up4 = UpSampling2D()(dec3)
    cat4 = concatenate([up4, c4], axis=3)
    dec4 = CBAR_block(cat4, fn_4)

    up5 = UpSampling2D()(dec4)
    cat5 = concatenate([up5, new_input], axis=3)
    dec5 = CBAR_block(cat5, fn_4)

    if num_classes == 1 or num_classes == 2:
        final_filter_num = 1
        final_activation = 'sigmoid'
    else:
        final_filter_num = num_classes
        final_activation = 'softmax'

    output = Conv2D(filters=final_filter_num, kernel_size=1, activation=final_activation)(dec5)
    model = keras.Model(inputs=new_input, outputs=output)
    print("Model Output Shape:", output.shape)
    return model

def load_image_dataset(image_dir, image_size, batch_size, subset):
    return tf.keras.utils.image_dataset_from_directory(
        image_dir,
        labels=None,
        color_mode='rgb',
        batch_size=batch_size,
        image_size=image_size,
        shuffle=True,
        seed=123,
        validation_split=0.2,
        subset=subset
    ).map(lambda x: x / 255.0)  # normalize

def load_mask_dataset(mask_dir, image_size, batch_size, subset, num_classes):
    ds = tf.keras.utils.image_dataset_from_directory(
        mask_dir,
        labels=None,
        color_mode='grayscale',
        batch_size=batch_size,
        image_size=image_size,
        shuffle=True,
        seed=123,
        validation_split=0.2,
        subset=subset
    )

    def preprocess_mask(x):
        x = tf.squeeze(x, axis=-1)  # shape: (B, H, W)
        x = tf.cast(x, tf.uint8)
        return tf.one_hot(x, depth=num_classes)

    return ds.map(lambda x: preprocess_mask(x))

def zip_dataset(image_ds, mask_ds):
    return tf.data.Dataset.zip((image_ds, mask_ds))

def train():
    epoch=10
    image_size = (224, 224)
    batch_size = 8
    num_classes = 58

    image_train = load_image_dataset('./image/', image_size, batch_size, 'training')
    image_val = load_image_dataset('./image', image_size, batch_size, 'validation')

    mask_train = load_mask_dataset('./mask', image_size, batch_size, 'training', num_classes)
    mask_val = load_mask_dataset('./mask', image_size, batch_size, 'validation', num_classes)

    train_ds = zip_dataset(image_train, mask_train)
    val_ds = zip_dataset(image_val, mask_val)

    model = efficient_unet(input_shape=(*image_size, 3), num_classes=num_classes)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(patience=2)
    ]

    model.fit(train_ds, validation_data=val_ds, epochs=epoch, callbacks=callbacks)
    model.save('efficient_unet_tfdata.h5')

if __name__ == '__main__':
    train()