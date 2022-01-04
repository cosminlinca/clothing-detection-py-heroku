import os

import numpy as np
import pydload
from keras_retinanet import models
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image


WEIGHTS_URL = 'https://github.com/notAi-tech/LogoDet/releases/download/292_classes_v1/weights'
CLASSES_URL = 'https://github.com/notAi-tech/LogoDet/releases/download/292_classes_v1/classes'

model_folder = os.path.join('yolo/weights')
if not os.path.exists(model_folder):
    os.mkdir(model_folder)

model_path = os.path.join(model_folder, 'weights')

if not os.path.exists(model_path):
    print('Downloading the checkpoint to', model_path)
    pydload.dload(WEIGHTS_URL, save_to_path=model_path, max_time=None)

classes_path = os.path.join(model_folder, 'classes')

if not os.path.exists(classes_path):
    print('Downloading the class list to', classes_path)
    pydload.dload(CLASSES_URL, save_to_path=classes_path, max_time=None)

detection_model = models.load_model(model_path, backbone_name='resnet50')
classes = open(classes_path).readlines()
classes = [i.strip() for i in classes if i.strip()]


def detect_single(img_path, min_prob=0.4):
    image = read_image_bgr(img_path)
    image = preprocess_image(image)
    image, scale = resize_image(image)
    boxes, scores, labels = detection_model.predict_on_batch(np.expand_dims(image, axis=0))
    boxes /= scale
    processed_boxes = []
    for box, score, label in zip(boxes[0], scores[0], labels[0]):
        if score < min_prob:
            continue
        # box = box.astype(int).tolist()
        # label = classes[label]
        processed_boxes.append(box.tolist()[0])
        processed_boxes.append(box.tolist()[1])
        processed_boxes.append(box.tolist()[2])
        processed_boxes.append(box.tolist()[3])
        processed_boxes.append(float(score))
        processed_boxes.append(label)

    return processed_boxes


def predictor(image_paths=[]):
    results = []
    for image_path in image_paths:
        try:
            results.append(detect_single(image_path))
        except Exception as ex:
            results.append([f'Failed with exception: {ex}'])

    return results


if __name__ == '__main__':
    import json
    import pickle
    import base64

    example = ["s-l300.jpg"]

    print(json.dumps(predictor(example)))

    example = {
        file_name: base64.b64encode(open(file_name, "rb").read()).decode("utf-8")
        for file_name in example
    }

    pickle.dump(example, open("example.pkl", "wb"), protocol=2)
