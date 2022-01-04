# import packages
import json
import os
# Imports for clothing detection
from datetime import datetime

from flask import Flask, request, redirect
from flask_cors import CORS, cross_origin
from werkzeug.utils import secure_filename

# Imports for brand detection
from .logo_predictor import predictor
from .wrapper import prepare_clothing_detection, track_result
from yolo.utils.utils import *

# Firebase
import pyrebase

config = {
    "apiKey": "AIzaSyDhfV5FpUKxwYe1aj6fKB66gSF5U0HpFzY",
    "authDomain": "clothingdetection.firebaseapp.com",
    "projectId": "clothingdetection",
    "databaseURL": "",
    "storageBucket": "clothingdetection.appspot.com",
    "serviceAccount": "serviceAccountKey.json"
}
firebase_storage = pyrebase.initialize_app(config)
storage = firebase_storage.storage()
UPLOAD_FOLDER = os.path.dirname(os.path.abspath(__file__))

ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'JPG', 'JPEG', 'PNG'}

application = Flask(__name__)
CORS(application)
DIR_PATH = os.path.dirname(os.path.realpath(__file__))
application.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# limit upload size upto 8mb
application.config['MAX_CONTENT_LENGTH'] = 8 * 1024 * 1024

# prepare ml model
arr = prepare_clothing_detection()


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def write_json(new_data, filename='data.json'):
    with open(filename, 'r+') as file:
        # First we load existing data into a dict.
        file_data = json.load(file)
        # Join new_data with file_data inside emp_details
        file_data["data"].append(new_data)
        # Sets file's current position at offset.
        file.seek(0)
        # convert back to json.
        json.dump(file_data, file, indent=4)


@application.route("/files", methods=['GET'])
@cross_origin()
def files():
    if request.method == 'GET':
        file_data = []
        with open('data.json', 'r') as file:
            # First we load existing data into a dict.
            file_data = json.load(file)["data"]

        return json.dumps(file_data)


@application.route("/upload", methods=['GET', 'POST'])
@cross_origin()
def index():
    if request.method == 'POST':
        json_prediction_result = {}

        if 'file' not in request.files:
            print('No file attached in request')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            print('No file selected')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            print(filename)
            path = "UPLOAD_FOLDER/" + filename
            if not os.path.isfile(path):
                file.save(path)
                while not os.path.exists(path):
                    time.sleep(1)
                storage.child(path).put(path)
                firebase_url = storage.child("UPLOAD_FOLDER/" + filename).get_url(None)
                print("URL: " + firebase_url)
                print("new path :", path)
            else:
                print("already exists: ", filename)
                file_data = []
                with open('data.json', 'r') as file:
                    # First we load existing data into a dict.
                    file_data = json.load(file)["data"]

                return json.dumps(file_data)

            json_prediction_result["file_path"] = firebase_url
            json_prediction_result["clothing_type"] = ""
            json_prediction_result["clothing_prediction"] = ""
            json_prediction_result["brand_type"] = ""
            json_prediction_result["brand_prediction"] = ""

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

            for x1, y1, x2, y2, cls_conf, cls_pred in clothing_predictions:
                track_result(tracking_file, cls_conf, cls_pred, path, classes, colors)
                json_prediction_result["clothing_type"] = classes[int(cls_pred)]
                json_prediction_result["clothing_prediction"] = cls_conf

            # Draw brand
            if len(brand_predictions[0]) != 0:
                brand_predictions.sort(reverse=False, key=lambda x: x[4])
                # Load classes for brand detection
                home = os.path.expanduser("~")
                model_folder = os.path.join(home, '.LogoDet/')
                if not os.path.exists(model_folder):
                    os.mkdir(model_folder)

                classes_path = os.path.join(model_folder, 'classes')
                classes = open(classes_path).readlines()
                classes = [i.strip() for i in classes if i.strip()]

                for x1, y1, x2, y2, cls_conf, cls_pred in brand_predictions:
                    track_result(tracking_file, cls_conf, cls_pred, path, classes, colors)
                    json_prediction_result["brand_type"] = classes[int(cls_pred)]
                    json_prediction_result["brand_prediction"] = cls_conf

            tracking_file.write("*****************************************")
            tracking_file.write("\n")
            tracking_file.seek(0)
            tracking_file.close()

            write_json(json_prediction_result)

            return json_prediction_result


if __name__ == "__main__":
    application.debug = True
    application.run(host='0.0.0.0')
