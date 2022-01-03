# Logo detection using LogoDet

### Use a venv with Python 3.6

### BRAND DETECTION
- YOLOv3 trained with Resnet50
- Dataset: LogoDet-3K: A Large-Scale Image Dataset for Logo Detection
- All weights and config files are downloaded
- Paper: LogoDet-3K: A Large-Scale Image Dataset for Logo Detection - https://arxiv.org/pdf/2008.05359.pdf

```python
# Simple example of usage
# Weights will be auto-downloaded
from predictor import predictor

image_paths = [image_1.jpg, image_2.png, ...]

predictions = predictor(image_paths)

```

**Notes about LogoDet**:

1. Although they were able to generate/ gather data for more than 2000 unique company logos, the current release is limited to [these 292 logos](https://github.com/notAI-tech/LogoDet/releases/download/292_classes_v1/classes.txt) due to hardware constraints.
