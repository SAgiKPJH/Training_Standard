import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import cv2
from keras.models import load_model
import traceback
import shutil

def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)

base_path = 'C:/Users/jh.park/Desktop/ADC_Edge'
path = base_path + '/Models'      #### 모델파일폴더를 지정 폴더에 담으면 모델 목록을 전부 keras 포멧으로 변환

result_root = base_path + '/Result_Model'      ## 결과 파일이 생성 될 root 폴더 경로

models = os.listdir(path)

print(f'[Log] Get Model List from: {models}')

for model_name in models : 
    print(f'[Log] Excute Model Save: [{model_name}]')
    step_name = os.path.splitext(model_name)[0]
    
    result_path = f'{result_root}/{step_name}'
    createFolder(result_path)

    keras_model = load_model(path + '/' + model_name)

    keras_model.save(f'{result_path}/{step_name}')

    flist = os.listdir(f'{result_path}/{step_name}')
    createFolder(f'{result_path}/{step_name}/1/')
    for file in flist :
        shutil.move(f'{result_path}/{step_name}/{file}', f'{result_path}/{step_name}/1/{file}')
    
    print(f'[Log] Model Save Complete: [{model_name}]')
