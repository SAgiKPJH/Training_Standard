import json
import os
import cv2
import pickle
import torch
import torchvision.transforms as transforms
from PIL import Image


class ModelHandler:
    def __init__(self, data, context):
        model_config_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model.json")
        with open(model_config_file_path, "r", encoding='utf-8') as config_file:
            __config = json.load(config_file)
            self.__model_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), __config['model_file'])

        self.__device = __config['device']
        extra_files = {'inference_info': '', 'label_info': ''}
        self.__model = torch.jit.load(self.__model_file_path, map_location=self.__device, _extra_files=extra_files).eval().to(self.__device)

        # extra_files 구조: {'label_info': '{"inference_info": "{...}"}', 'inference_info': '{}'}
        # 실제 데이터는 label_info 키 안에 inference_info로 중첩
        label_info_raw = json.loads(extra_files['label_info'].decode('utf-8') if isinstance(extra_files['label_info'], bytes) else extra_files['label_info'])
        inner = label_info_raw.get('inference_info', '{}')
        if isinstance(inner, str):
            inner = json.loads(inner)
        self.__label_info = inner['label_info']
        self.__input_size = inner['input_size']

        self.__transform = transforms.Compose([
            transforms.Lambda(lambda img: Image.fromarray(img).convert("RGB")),
            transforms.Resize((self.__input_size, self.__input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def __call__(self, data, context):
        data = pickle.loads(data)

        if data.ndim == 2:
            data = cv2.cvtColor(data, cv2.COLOR_GRAY2BGR)

        # BGR 그대로 전달 (학습 시 BGR로 학습됨)
        data = self.__transform(data)
        data = data.unsqueeze(0).to(self.__device)

        with torch.no_grad():
            predict = self.__model(data)
            if isinstance(predict, tuple):
                predict = predict[0]
            predict = predict.to('cpu')
            predict = torch.nn.functional.softmax(predict, dim=1)

        output = list()
        for i in range(self.__label_info['label_count']):
            current_label = dict(self.__label_info[f'label_{i}'])
            current_label['score'] = predict[0][i].item()
            output.append(current_label)

        output = sorted(output, key=lambda x: x['score'], reverse=True)

        return pickle.dumps(output), context


if __name__ == "__main__":
    import cv2
    test_data_path = "data/image_line.png"
    test_data = cv2.imread(test_data_path, 1)
    test_data_pickle = pickle.dumps(test_data)

    # =================================================================== #

    handler = ModelHandler(None, None)
    inference_result, context = handler(test_data_pickle, None)

    # =================================================================== #

    inference_result = pickle.loads(inference_result)
    print(*inference_result, sep="\n")
