import cv2
import os
import pickle
import numpy as np
import tensorflow as tf

# GPU 메모리 제한
memory_limit = 3  # GB 기준
physical_gpus = tf.config.list_physical_devices('GPU')
if physical_gpus:
    try:
        for gpu in physical_gpus:
            tf.config.experimental.set_virtual_device_configuration(
                gpu, [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=memory_limit * 1024)]
            )
    except RuntimeError as e:
        print(e)
        pass

class ModelHandler:
    def __init__(self, data=None, context=None):
        base_path = os.path.dirname(os.path.abspath(__file__))

        # YOLO 설정 파일, 가중치, 클래스 정보 경로 설정
        self.weights_path = os.path.join(base_path, "data", "yolov4.weights")
        self.config_path = os.path.join(base_path, "data", "yolov4.cfg")
        self.names_path = os.path.join(base_path, "data", "coco.data")

        # 클래스 이름 로드
        with open(self.names_path, 'r') as f:
            self.class_names = [line.strip() for line in f.readlines()]

        # 네트워크 로드
        self.net = cv2.dnn.readNetFromDarknet(self.config_path, self.weights_path)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

        self.conf_threshold = 0.25
        self.nms_threshold = 0.4
        self.input_size = (416, 416)

    def __call__(self, data, context):
        frame = pickle.loads(data)  # numpy array

        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, self.input_size, swapRB=True, crop=False)
        self.net.setInput(blob)

        # 출력 레이어 이름
        layer_names = self.net.getLayerNames()
        output_layers = [layer_names[i - 1] for i in self.net.getUnconnectedOutLayers().flatten()]

        # 추론
        outputs = self.net.forward(output_layers)

        height, width = frame.shape[:2]
        boxes, confidences, class_ids = [], [], []

        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = int(np.argmax(scores))
                confidence = scores[class_id]
                if confidence > self.conf_threshold:
                    center_x, center_y, w, h = (detection[0:4] * [width, height, width, height]).astype("int")
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    boxes.append([x, y, int(w), int(h)])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        # NMS 적용
        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.conf_threshold, self.nms_threshold)

        result = {
            "detections": [],
            "image": None
        }

        for i in indices.flatten():
            x, y, w, h = boxes[i]
            label = class_ids[i]
            confidence = float(confidences[i])

            result["detections"].append({
                "label": label,
                "confidence": confidence,
                "box": [x, y, w, h]
            })

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} {confidence:.2f}", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        result["image"] = frame

        return pickle.dumps(result), context

if __name__ == '__main__':
    handler = ModelHandler()

    image_path = "test50.jpg"
    image = cv2.imread(image_path)
    data = pickle.dumps(image)

    output, _ = handler(data)
    result = pickle.loads(output)