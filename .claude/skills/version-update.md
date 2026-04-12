---
name: version-update
description: 버전 업데이트 절차
---

## 버전 업데이트

버전 문자열이 있는 파일 6개를 모두 업데이트해야 합니다:

```bash
grep -rn "v1\." "Training Standard/" --include="*.py" | grep version
```

### 대상 파일
1. Pytorch_CNN_Standard/pytorch_classification_local.py
2. Pytorch_CNN_Standard/pytorch_classification_trainstandard.py
3. Tensorflow_CNN_Standard/tensorflow_classification_local.py
4. Tensorflow_CNN_Standard/tensorflow_classification_trainstandard.py
5. CNN_Training_Standard/classification_training_standard.py
6. CNN_Training_Standard/classification_training_standard_local.py

### 형식
- `Pytorch_CNN_Standard_v{major}.{minor}.{patch}`
- `Tensorflow_CNN_Standard_v{major}.{minor}.{patch}`
- `CNN_Training_Standard_v{major}.{minor}.{patch}`
