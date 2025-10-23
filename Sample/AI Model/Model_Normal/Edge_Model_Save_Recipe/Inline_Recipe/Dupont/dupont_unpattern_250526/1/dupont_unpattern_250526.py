from cmath import inf
import cv2
import numpy as np
from exception import RecipeException
from typing import Dict
from inference import Inference
import sys, os


def label_change(predict,label_list) :
    result = label_list[np.argmax(predict)]
    top_score = np.max(predict)
    return result, top_score

def Run(images:Dict[str, str], inference:Inference):
    try:
        step = 'UnPattern_S_C_step_250519'                ### 레시피, 모델명
        #Th = 0.5
        ## Label list
        label_list = ['3100','3101','3102','3103','3104','3105','3106','3200','3300','3400','3600','3700']

        img = cv2.resize(cv2.imread(images['0'],1),(256,256))  # Left High
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        ## Voting Logic
        pre_result = inference.predict(model_name=f'{step}_weight', image=img)
        # mixup_result = inference.predict(model_name=f'{step}_mixup', image=img)
        # voting_result = (pre_result + mixup_result) /2
        # result,top_score = label_change(voting_result,label_list)
        result,top_score = label_change(pre_result,label_list)
        result = int(str(result)[:2])
        ## Threshold Logic
        #if top_score < Th :
        #    result = '231'
        inference.result(int(result))
        

    except Exception as e:
        raise RecipeException(inference=inference, exception=e)