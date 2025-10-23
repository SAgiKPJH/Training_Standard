```shell
python --version
python -m venv .venv
.\.venv\Scripts\Activate.ps1

pip install tensorflow==2.12.0 opencv-python keras matplotlib

pip install -r ./TensorFlow/requirements.txt
pip install -r ./Pytorch/requirements.txt

python .\Edge_Model_Save_Recipe\Model_Save\model_save.py

python .\Edge_Model_Save_Recipe\Model_Validation\validateion.py
```