import os
import pickle
import numpy as np
import torch
import torchvision.transforms as transforms
from keras.models import *
import cv2

import tensorflow as tf
import json

## Gpu Memory 제한
memory_limit = 3
gpus = tf.config.list_physical_devices('GPU')
if gpus :
    try : 
        for gpu in gpus :
            tf.config.experimental.set_virtual_device_configuration(gpu,[tf.config.experimental.VirtualDeviceConfiguration(memory_limit=memory_limit*1024)])
    except RuntimeError as e :
        print(e)
        pass


class ModelHandler:
    def __init__(self, data, context):
        '''
        Json Sample이 없어서 Skip하고 임의로 코딩해서 배포 하겠습니다.
        config_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model.json")

        with open(config_file_path, "r", encoding='utf-8') as config_file:
            config = json.load(config_file)
            model_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), config['model_file'])
        '''
        self.model = load_model(os.path.join(os.path.dirname(os.path.abspath(__file__)),"model.h5"))

    def test(self) :        ## 이후에 추론 하는 모든 데이터에 대해서 빠르게 진행 될 수 있도록 하는 부분
        first_layer = self.model.layers[0]
        input_shape = first_layer.input_shape[0][1:]
        data = np.zeros(input_shape)    # input_shape[0],input_shape[1],3
        data = np.expand_dims(data,0)   ## 1,input_shape[0],input_shape[1],3
        self.model.predict(data,batch_size=1)

    def __call__(self, data, context):
        ## Expand_dims
        data = pickle.loads(data)
        images = np.array(data) # input_shape[0],input_shape[1],3
        images = np.expand_dims(images,0) ## 1,input_shape[0],input_shape[1],3
        result = {}
        
        ## Label Data Load
        label = open(os.path.join(os.path.dirname(os.path.abspath(__file__)),"label.txt"),'r')
        label_list = label.read().replace('\n','')
        label_list = label_list.split(',')

        ## Predict
        predict_result = self.model.predict(images)
        adc_code = label_list[np.argmax(predict_result)]
        top1_score = np.max(predict_result)
        result['result_code'] = adc_code
        result['score'] = top1_score
        result = pickle.dumps(result)
        return result, context
        
if __name__ == "__main__":
    img = cv2.imread(os.path.join(r'D:\ADC60_Trainset\01','2_000_MCP20503A00-004_MS2360033-02A_00-39.png'))
    img = cv2.resize(img,(224,224))
    data = pickle.dumps(img)
    ModelHandler(None, None).__call__(data=data,context={})