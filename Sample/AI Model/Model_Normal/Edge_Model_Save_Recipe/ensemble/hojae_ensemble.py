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
        step = 'COF'                ### 레시피, 모델명
        Th = 0.5
        ## Label list
        label_list = ['0','1']
        label_list2 = ['2','3','4']
        img = cv2.resize(cv2.imread(images['11'],1),(256,256))  # Left High
        img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
        ## Voting Logic
        pre_result = inference.predict(model_name=f'{step}_pre', image=img)     ## 1번째 로직 모델
        # mixup_result = inference.predict(model_name=f'{step}_mixup', image=img)
        # voting_result = (pre_result + mixup_result) /2
        # result,top_score = label_change(voting_result,label_list)

        result,top_score = label_change(pre_result,label_list)
        if result == '0' :       ## false
            pre_result2 = inference.predict(model_name=f'{step}_pre2', image=img)    ## 2번째 로직 모델
            # mixup_result2 = inference.predict(model_name=f'{step}_mixup2', image=img)    ## 2번째 로직 모델
            # voting_result2 = (pre_result2 + mixup_result2) /2
            # result,top_score = label_change(voting_result2,label_list2)
            result,top_score = label_change(pre_result2,label_list2)
        ## Threshold Logic
        if top_score < Th :
            result = '231'
        inference.result(int(result))
        

    except Exception as e:
        raise RecipeException(inference=inference, exception=e)