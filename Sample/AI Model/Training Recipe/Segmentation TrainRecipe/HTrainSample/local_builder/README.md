## 🧱 1. 가상환경 생성 및 활성화

```bash
# Python 3.10 기준
python -m venv unet-env

# Windows
unet-env\Scripts\activate

# macOS/Linux
source unet-env/bin/activate
```

---

## 📦 2. 필수 패키지 설치

### 🔹 PyPI 설치

```bash
pip install --upgrade pip

pip install tensorflow==2.12.0 keras==2.12.0
pip install numpy matplotlib opencv-python
```

> ✅ `EfficientNetB3`는 `keras.applications.efficientnet`에 내장되어 있으므로 추가 설치는 필요 없습니다.

---

## 📄 3. requirements.txt 사용 시

`requirements.txt` 파일로 관리하고 싶으시다면, 다음 내용을 저장한 후:

```txt
tensorflow==2.12.0
keras==2.12.0
numpy
matplotlib
opencv-python
```

설치는 다음 명령어로 가능:

```bash
pip install -r requirements.txt
```

---

## 🧪 설치 확인용 코드

가상환경이 정상 구성되었는지 확인하려면 아래 코드를 실행해보세요:

```python
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB3

model = EfficientNetB3(weights=None, input_shape=(224, 224, 3), include_top=False)
print("EfficientNetB3 loaded successfully")
```