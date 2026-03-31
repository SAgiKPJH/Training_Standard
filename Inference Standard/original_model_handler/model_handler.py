import json
import os
import cv2
import pickle
import torch
import torchvision.transforms as transforms


class ModelHandler:
    def __init__(self, data, context):
        model_config_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model.json")
        with open(model_config_file_path, "r", encoding='utf-8') as config_file:
            __config = json.load(config_file)
            self.__model_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), __config['model_file'])

        self.__device = __config['device']
        extra_files = {'inference_info' : {}}
        self.__model = torch.jit.load(self.__model_file_path, map_location=self.__device, _extra_files=extra_files).eval().to(self.__device)
        #self.__model = torch.jit.optimize_for_inference(torch.jit.script(self.__model))

        inference_info = json.loads(extra_files['inference_info'].decode('ascii'))
        self.__label_info = inference_info['label_info']
        self.__input_size = inference_info['input_size']

        self.__transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((self.__input_size, self.__input_size)),
            transforms.Nomalize(mean=[0.5, 0.5, 0.5], std=[0.5,0.5,0.5])
        ])

    def __call__(self, data, context):
        data = pickle.loads(data)

        if data.ndim == 2: data = cv2.cvtColor(data, cv2.COLOR_GRAY2BGR)

        data = self.__transform(data)
        data = data.unsqueeze(0).to(self.__device)
        
        with torch.no_grad():
            predict = self.__model(data)
            predict = predict.to('cpu')
            predict = torch.nn.functional.softmax(predict, dim=1)
        
        output = list()
        for i in range(self.__label_info['label_count']):
            current_label = self.__label_info[f'label_{i}']
            current_label['score'] = predict[0][i].item()
            output.append(current_label)
        
        output = sorted(output, key = lambda x : x['score'], reverse=True)

        return pickle.dumps(output), context

if __name__ == "__main__":
    
    import cv2
    test_data_path = "image.png"
    test_data = cv2.imread(test_data_path, 1)
    test_data = cv2.cvtColor(test_data, cv2.COLOR_BGR2GRAY)
    test_data_pickle = pickle.dumps(test_data)

    # =================================================================== #

    handler = ModelHandler(None, None)
    inference_result, context =  handler(test_data_pickle, None)

    # =================================================================== #

    inference_result = pickle.loads(inference_result)
    print(*inference_result, sep="\n")