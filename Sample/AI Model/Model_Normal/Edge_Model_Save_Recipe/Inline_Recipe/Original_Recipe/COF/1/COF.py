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
        #Th = 0.5
        ## Label list
        label_list = ['1000', '1001', '1002', '1003', '1004', '1005', '1006', '1007', '1008', '1009', '1010', '3000', '3001', '3002', '3003', '3004', '3005', '3006', '3007', '3008', '3009', '3010', '3100', '3100_1', '3101', '3102', '3103', '3103_1', '3104', '3105', '3106', '3107', '3108', '3109', '3110', '3200', '3201', '3202', '3203', '3204', '3205', '3206', '3207', '3208', '3209', '3300', '3301', '3302', '3303', '3304', '3305', '3306', '3307', '3308', '3309', '3310', '3400', '3401', '3403', '3405', '3406', '3407', '3408', '3409', '3410', '3500', '3501', '3502', '3503', '3504', '3505', '3506', '3507', '3508', 
'3509', '3510', '3600', '3603', '3604', '3605', '3606', '3607', '3608', '3609', '3610', '3700', '3702', '3703', '3705', '3706', '3707', '3709', '3801', '3900', '3903', '3904', '3905', '3906', '3907', '3908', '3909', '4000', '4003', '4006', '4008', '4009', '4101', '4103', '4107', '4200', '4201', '4203', '4204', '4205', '4206', '4207', '4208', '4209', '4210', '4300', '4302', '4303', '4305', '4306', '4308', '4906', '4907', '4909']
        
        img = cv2.resize(cv2.imread(images['0'],1),(256,256))  # Left High
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        ## Voting Logic
        pre_result = inference.predict(model_name=f'{step}_pre', image=img)
        # mixup_result = inference.predict(model_name=f'{step}_mixup', image=img)
        # voting_result = (pre_result + mixup_result) /2
        # result,top_score = label_change(voting_result,label_list) 
        result,top_score = label_change(pre_result,label_list)
        
        ADC_Code = result
        
        if ADC_Code == "1000" or ADC_Code == "1001" or ADC_Code == "1002" or ADC_Code == "1003" or ADC_Code == "1004" or ADC_Code == "1005" or ADC_Code == "1006" or ADC_Code == "1007" or ADC_Code == "1008" or ADC_Code == "1009" or ADC_Code == "1010" :
            result = "1"
                
        if ADC_Code == "3000" or ADC_Code == "3001" or ADC_Code == "3002" or ADC_Code == "3003" or ADC_Code == "3004" or ADC_Code == "3005" or ADC_Code == "3006" or ADC_Code == "3007" or ADC_Code == "3008" or ADC_Code == "3009" or ADC_Code == "3010":
            result = "30"

        if ADC_Code == "3100" or ADC_Code == "3100_1" or ADC_Code == "3101" or ADC_Code == "3102" or ADC_Code == "3103" or ADC_Code == "3103_1" or ADC_Code == "3104" or ADC_Code == "3105" or ADC_Code == "3106" or ADC_Code == "3107" or ADC_Code == "3108" or ADC_Code == "3109" or ADC_Code == "3110":
            result = "31"

        if ADC_Code == "3200" or ADC_Code == "3201" or ADC_Code == "3202" or ADC_Code == "3203" or ADC_Code == "3204" or ADC_Code == "3205" or ADC_Code == "3206" or ADC_Code == "3207" or ADC_Code == "3208" or ADC_Code == "3209":
            result = "32"

        if ADC_Code == "3300" or ADC_Code == "3301" or ADC_Code == "3302" or ADC_Code == "3303" or ADC_Code == "3304" or ADC_Code == "3305" or ADC_Code == "3306" or ADC_Code == "3307" or ADC_Code == "3308" or ADC_Code == "3309" or ADC_Code == "3310":
            result = "33"

        if ADC_Code == "3400" or ADC_Code == "3401" or ADC_Code == "3403" or ADC_Code == "3405" or ADC_Code == "3406" or ADC_Code == "3407" or ADC_Code == "3408" or ADC_Code == "3409" or ADC_Code == "3410":
            result = "34"

        if ADC_Code == "3500" or ADC_Code == "3501" or ADC_Code == "3502" or ADC_Code == "3503" or ADC_Code == "3504" or ADC_Code == "3505" or ADC_Code == "3506" or ADC_Code == "3507" or ADC_Code == "3508" or ADC_Code == "3509" or ADC_Code == "3510":
            result = "35"

        if ADC_Code == "3600" or ADC_Code == "3603" or ADC_Code == "3604" or ADC_Code == "3605" or ADC_Code == "3606" or ADC_Code == "3607" or ADC_Code == "3608" or ADC_Code == "3609" or ADC_Code == "3610":
            result = "36"

        if ADC_Code == "3700" or ADC_Code == "3702" or ADC_Code == "3703" or ADC_Code == "3705" or ADC_Code == "3706" or ADC_Code == "3707" or ADC_Code == "3709":
            result = "37"
        
        if ADC_Code == "3801" :
            result = "38"
        
        if ADC_Code == "3900" or ADC_Code == "3903" or ADC_Code == "3904" or ADC_Code == "3905" or ADC_Code == "3906" or ADC_Code == "3907" or ADC_Code == "3908" or ADC_Code == "3909":
            result = "39"
        
        if ADC_Code == "4000" or ADC_Code == "4003" or ADC_Code == "4006" or ADC_Code == "4008" or ADC_Code == "4009":
            result = "40"
            
        if ADC_Code == "4101" or ADC_Code == "4103" or ADC_Code == "4107" :
            result = "41"
        
        if ADC_Code == "4200" or ADC_Code == "4201" or ADC_Code == "4203" or ADC_Code == "4204" or ADC_Code == "4205" or ADC_Code == "4206" or ADC_Code == "4207" or ADC_Code == "4208" or ADC_Code == "4209" or ADC_Code == "4210":
            result = "42"

        if ADC_Code == "4300" or ADC_Code == "4302" or ADC_Code == "4303" or ADC_Code == "4305" or ADC_Code == "4306" or ADC_Code == "4308" :
            result = "43"

        if ADC_Code == "4906" or ADC_Code == "4907" or ADC_Code == "4909":
            result = "49"
        
        
        ## Threshold Logic
        #if top_score < Th :
        #    result = '231'
        inference.result(int(result))
        

    except Exception as e:
        raise RecipeException(inference=inference, exception=e)