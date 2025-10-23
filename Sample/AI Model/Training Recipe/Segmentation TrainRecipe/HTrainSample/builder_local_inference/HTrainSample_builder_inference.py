# python .\HTrainSample_builder_inference.py

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.utils import load_img, img_to_array
import matplotlib.pyplot as plt

label_map = {
    0: 2,
    1: 4,
    2: 10,
    3: 20,
    4: 50,
    5: 100,
    6: 150,
    7: 200,
    8: 220,
    9: 255
}

def load_model(model_path):
    model = keras.models.load_model(model_path)
    return model

def preprocess_image(image_path, target_size=(224, 224)):
    img = load_img(image_path, target_size=target_size)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # 배치 차원 추가
    return img_array

def run_inference(model, image_tensor):
    prediction = model.predict(image_tensor)
    return prediction

def visualize_prediction(image_tensor, prediction):
    image_np = np.squeeze(image_tensor).astype(np.uint8)  # (224, 224, 3)
    
    pred_indices = np.argmax(prediction[0], axis=-1)  # (224, 224)
    label_mask = np.vectorize(label_map.get)(pred_indices).astype(np.uint8)

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(image_np)
    ax[0].set_title('Input Image')
    ax[0].axis('off')

    ax[1].imshow(label_mask, cmap='gray', vmin=0, vmax=255)
    ax[1].set_title('Predicted Label Mask')
    ax[1].axis('off')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    model_path = "./model_3.h5"
    image_path = "./test_image2.png"

    model = load_model(model_path)
    image_tensor = preprocess_image(image_path)
    prediction = run_inference(model, image_tensor)
    visualize_prediction(image_tensor, prediction)