import pickle
import cv2
import numpy as np
import logging
from os import path

from segment_anything import sam_model_registry, SamPredictor
from segment_anything import predictor
import torch
import cv2

logger = logging.getLogger()

class ModelHandler:
    def __init__(self, data, context):
        
        self.__device = "cpu"
        __extra_files = {'inference_info' : {}}

        sam_checkpoint = path.join(path.dirname( path.abspath(__file__) ), "segment_anything", "sam_vit_b_01ec64.pth")
        model_type = "vit_b"

        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        sam.to(device=self.__device)

        self.__model = SamPredictor(sam)

    def __call__(self, data, context):
        data = pickle.loads(data)

        if data.ndim == 3:
            gray_image = cv2.cvtColor(data, cv2.COLOR_BGR2GRAY)
            rgb_image = data
        elif data.ndim == 2:
            gray_image = data
            rgb_image = cv2.cvtColor(data, cv2.COLOR_GRAY2BGR)

        self.__model.set_image(rgb_image)

        # _, binary_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)
        
        # binary_image = cv2.bitwise_not(binary_image)
        
        # nz = cv2.findNonZero(binary_image)

        rois = {'1':{'box':[32,38,40,41]},'2':{'box':[80,41,34,40]},'3':{'box':[93,90,35,37]},'4':{'box':[46,89,38,38]},'5':{'box':[4,86,33,42]},'6':{'box':[138,90,37,39]}}

        logger.info(f"rois")
        roi_list = list()
        logger.info(f"roi_list")
        for key in rois:
            logger.info(f"key: {key}")
            logger.info(f"roi[key]: {rois[key]}")

            roi = rois[key]['box']
            logger.info(f"key {key}: {roi}")
            roi_list.append(roi)
            logger.info(f"roi_list append({key})")
        logger.info(f"roi_list: {roi_list}")
        input_boxes = torch.tensor(roi_list, device=self.__device)
        logger.info(f"input_boxes: {input_boxes}")
        transformed_boxes = self.__model.transform.apply_boxes_torch(input_boxes, rgb_image.shape[:2])
        logger.info(f"transformed_boxes: {transformed_boxes}")
        # input_point = np.array([[55, 64]])
        # input_label = np.array([1])
        # input_box = np.array([207, 1366, 1468, 1757])

        # masks, scores, logits = self.__model.predict(
        #     point_coords=input_point,
        #     point_labels=input_label,
        #     multimask_output=False,
        # )

        # masks, scores, logits = self.__model.predict(
        #     point_coords=None,
        #     point_labels=None,
        #     box=input_box[None, :],
        #     multimask_output=False,
        # )

        masks, scores, _ = self.__model.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes,
            multimask_output=False,
        )

        output = list()
        for i, (mask, score) in enumerate(zip(masks, scores)):
            numpy_test = mask.cpu().numpy()

            logger.info(f"mask.ahpe: {numpy_test.shape}")
            second_numpy = np.array(mask)
            logger.info(f"second shape: {second_numpy.shape}")
            output.append((mask.cpu().numpy().astype('uint8')*255, score))


        return pickle.dumps(output), context

if __name__ == '__main__':

    import cv2
    
    test_data_path = "..\\..\\..\\test\\testdata\\input_data\\noisy\\500x500_lena_gaussian_20_noise_1.png"
    test_data = cv2.imread(test_data_path, 1)
    test_data_pickle = pickle.dumps(test_data)
  
    # =================================================================== #

    handler = ModelHandler(None, None)
    inference_result, context =  handler(test_data_pickle, None)

    # =================================================================== #

    inference_result = pickle.loads(inference_result)
    cv2.imwrite("..\\dncnn\\inference_result.png", inference_result)
    print("Success")