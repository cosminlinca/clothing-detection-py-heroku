# Imports for clothing detection
from datetime import datetime

from predictors.YOLOv3 import YOLOv3Predictor
from yolo.utils.utils import *
import gdown


# Imports for brand detection
from logo_predictor import predictor
import os
import numpy as np

def prepare_clothing_detection():
    global yolo_params
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device.type)
    torch.cuda.empty_cache()

    url = 'https://drive.google.com/uc?id=1Ur-YLA3awzZqOqPppvnUJRY6dwENwj0S'
    output = 'yolo/weights/yolov3-df2_15000.weights'
    gdown.download(url, output, quiet=False)

    # YOLO PARAMS
    yolo_df2_params = {"model_def": "yolo/df2cfg/yolov3-df2.cfg",
                       "weights_path" : "yolo/weights/yolov3-df2_15000.weights",
                       "class_path": "yolo/df2cfg/df2.names",
                       "conf_thres": 0.8,
                       "nms_thres": 0.4,
                       "img_size": 416,
                       "device": device}

    yolo_modanet_params = {"model_def": "yolo/modanetcfg/yolov3-modanet.cfg",
                           "weights_path" : "yolo/weights/yolov3-modanet_last.weights",
                           "class_path": "yolo/modanetcfg/modanet.names",
                           "conf_thres": 0.5,
                           "nms_thres": 0.4,
                           "img_size": 416,
                           "device": device}

    # DATASET
    dataset = 'df2'

    if dataset == 'df2':  # deepfashion2
        yolo_params = yolo_df2_params

    if dataset == 'modanet':
        yolo_params = yolo_modanet_params

    # Classes
    classes = load_classes(yolo_params["class_path"])

    # Colors
    cmap = plt.get_cmap("rainbow")
    colors = np.array([cmap(i) for i in np.linspace(0, 1, 13)])

    return [YOLOv3Predictor(params=yolo_params), classes, colors]


def track_result(tracking_file, cls_conf, cls_pred, path, classes, colors):
    tracking_file.write("\t+ Label: %s, Conf: %.5f" % (classes[int(cls_pred)], cls_conf))
    tracking_file.write("\n")


def draw_type_box(x1, y1, x2, y2, cls_conf, cls_pred, img, classes, colors):
    print("\t+ Label: %s, Conf: %.5f" % (classes[int(cls_pred)], cls_conf))
    color = colors[int(cls_pred) % 12]
    color = tuple(c * 255 for c in color)
    color = (.7 * color[2], .7 * color[1], .7 * color[0])
    font = cv2.FONT_HERSHEY_SIMPLEX
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    text = "%s conf: %.3f" % (classes[int(cls_pred)], cls_conf)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
    y1 = 0 if y1 < 0 else y1
    y1_rect = y1 - 25
    y1_text = y1 - 5
    if y1_rect < 0:
        y1_rect = y1 + 27
        y1_text = y1 + 20
    cv2.rectangle(img, (x1 - 2, y1_rect), (x1 + int(8.5 * len(text)), y1), color, -1)
    cv2.putText(img, text, (x1, y1_text), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)


if __name__ == '__main__':
    arr = prepare_clothing_detection()
    clothing_model = arr[0]
    classes = arr[1]
    colors = arr[2]

    correctAnswers = 0
    wrongAnswers = 0

    for filename in os.listdir("T-Shirts-women"):
        path = os.path.join("T-Shirts-women", filename)
        print(filename)

        clothing_model = arr[0]
        classes = arr[1]
        colors = arr[2]

        # Open tracking file
        tracking_file = open('tracking.txt', "a+")
        tracking_file.write(path)
        tracking_file.write("\n")
        tracking_file.write(datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
        tracking_file.write("\n")

        # Load img
        img = cv2.imread(path)
        # Detect type of clothing
        clothing_predictions = clothing_model.get_detections(img)
        # Detect brand of clothing
        brand_predictions = predictor([path])

        print(clothing_predictions)
        print(brand_predictions)

        # Draw type
        if len(clothing_predictions) != 0:
            correctAnswers = correctAnswers + 1
            clothing_predictions.sort(reverse=False, key=lambda x: x[4])
            for x1, y1, x2, y2, cls_conf, cls_pred in clothing_predictions:
                draw_type_box(x1, y1, x2, y2, cls_conf, cls_pred, img, classes, colors)
                track_result(tracking_file, cls_conf, cls_pred, path, classes, colors)
        else:
            wrongAnswers = wrongAnswers + 1

        # Draw brand
        # if len(brand_predictions[0]) != 0:
        #     brand_predictions.sort(reverse=False, key=lambda x: x[4])
        #     # Load classes for brand detection
        #     home = os.path.expanduser("~")
        #     model_folder = os.path.join(home, '.LogoDet/')
        #     if not os.path.exists(model_folder):
        #         os.mkdir(model_folder)
        #
        #     model_path = os.path.join(model_folder, 'weights')
        #     classes_path = os.path.join(model_folder, 'classes')
        #     classes = open(classes_path).readlines()
        #     classes = [i.strip() for i in classes if i.strip()]
        #
        #     for x1, y1, x2, y2, cls_conf, cls_pred in brand_predictions:
        #         draw_type_box(x1, y1, x2, y2, cls_conf, cls_pred, img, classes, colors)
        #         track_result(tracking_file, cls_conf, cls_pred, path, classes, colors)

        tracking_file.write("*****************************************")
        tracking_file.write("\n")
        tracking_file.seek(0)
        tracking_file.close()

        # Reverse colors
        b, g, r = cv2.split(img)  # get b,g,r
        img = cv2.merge([r, g, b])  # switch it to rgb
        # Display final image
        # plt.imshow(img, cmap='spring')
        # plt.gcf().set_dpi(600)
        # plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
        # plt.show()

    print("Correct predictions:{0}".format(str(correctAnswers)))
    print("Wrong:{0}".format(str(wrongAnswers)))
