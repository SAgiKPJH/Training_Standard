import os
import cv2
import numpy as np
from keras.models import load_model

model_name = 'UnPattern_S_C_step_250519_model.pb'
model_path = 'C:/Users/jh.park/Desktop/ADC_Edge/Result_Model/UnPattern_S_C_step_250519_model/UnPattern_S_C_step_250519_model/1'  # 모델 파일 경로를 지정하세요
image_path = 'C:/Users/jh.park/Desktop/ADC_Edge/Edge_Model_Save_Recipe/Model_Validation/image.png'  # 테스트 이미지 경로를 지정하세요
save_path = 'C:/Users/jh.park/Desktop/ADC_Edge/Model_Validation_Result'  # 테스트 이미지 경로를 지정하세요

def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)

# 1. 모델 로딩
model = load_model(model_path)

# 2. 테스트 이미지 로딩
img = cv2.imread(image_path, cv2.IMREAD_COLOR)
img = cv2.resize(img, (256, 256))  # 모델 입력 크기에 맞춤
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_input = img.astype(np.float32) / 255.0  # 정규화
img_input = np.expand_dims(img_input, axis=0)   # (1, 256, 256)
img_input = np.expand_dims(img_input, axis=-1)  # (1, 256, 256, 1)

# 3. 추론
pred = model.predict(img_input)

# 4. 결과 시각화
import matplotlib.pyplot as plt
import numpy as np

# 예측 벡터 (예: softmax 결과)
pred = model.predict(img_input)
pred_vector = pred[0]  # (예: [0.05, 0.15, ..., 0.03])

# 자동으로 x축 인덱스 생성
x = np.arange(len(pred_vector))
labels = [str(i) for i in x]  # ['0', '1', '2', ..., 'n']

# 막대그래프 그리기
plt.figure(figsize=(10, 4))
plt.bar(x, pred_vector)
plt.xticks(x, labels)
plt.xlabel("Class Index")
plt.ylabel("Probability")
plt.title("Model Prediction (auto-class labels)")
plt.ylim(0, 1)
plt.grid(True)
plt.tight_layout()

createFolder(save_path)
plt.savefig(f'{save_path}/{model_name}.png')
print(f"저장 완료: {model_name}")

plt.show()