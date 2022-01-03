# BRAND DETECTION
# - YOLOv3 trained with Resnet50
# - Dataset: LogoDet-3K: A Large-Scale Image Dataset for Logo Detection
# - All weights and config files are downloaded
# - Paper: LogoDet-3K: A Large-Scale Image Dataset for Logo Detection - https://arxiv.org/pdf/2008.05359.pdf

from logo_predictor import predictor

image_paths = ["tests/0001test.jpg", "tests/test06.jpg", "tests/test07.jpg"]

predictions = predictor(image_paths)

for pred in predictions:
    print(pred)

