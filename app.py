import os
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename
from config import Config
from projects.object_detection import model
from hub import allow_file
import tensorflow as tf
from PIL import Image
import numpy as np
from projects.object_detection.models.utils import visualization_utils


app = Flask(__name__)

# all the pages
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/home")
def home():
    return render_template("index.html")

@app.route("/about")
def about():
    return render_template("index.html")

@app.route('/services#<section_id>')
def services(section_id):
    return render_template('index.html', section_id=section_id)


@app.route("/portfolio")
def portfolio():
    return render_template("portfolio.html")

@app.route("/contact")
def contact():
    return render_template("index.html")


# object_detect

@app.route('/image_detect')
def image_detect():
    return render_template('image_detect.html')


@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['file']
    if file and allow_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(Config.OBJ_UPLOAD_FOLDER, filename))
        return redirect(url_for('uploaded_file',filename=filename))


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    PATH_TO_TEST_IMAGES_DIR = Config.OBJ_UPLOAD_FOLDER
    TEST_IMAGE_PATHS = [os.path.join(PATH_TO_TEST_IMAGES_DIR,filename.format(i)) for i in range(1, 2) ]
    IMAGE_SIZE = (12, 8)

    detection_graph, category_index = model.init_detection()

    with detection_graph.as_default():
        with tf.compat.v1.Session(graph=detection_graph) as sess:
            for image_path in TEST_IMAGE_PATHS:
                image = Image.open(image_path)
                image_np = model.convert_image_to_array(image)
                image_np_expanded = np.expand_dims(image_np, axis=0)
                image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
                boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
                scores = detection_graph.get_tensor_by_name('detection_scores:0')
                classes = detection_graph.get_tensor_by_name('detection_classes:0')
                num_detections = detection_graph.get_tensor_by_name('num_detections:0')
                (boxes, scores, classes, num_detections) = sess.run(
                    [boxes, scores, classes, num_detections],
                    feed_dict={image_tensor: image_np_expanded})
                visualization_utils.visualize_boxes_and_labels_on_image_array(
                    image_np,
                    np.squeeze(boxes),
                    np.squeeze(classes).astype(np.int32),
                    np.squeeze(scores),
                    category_index,
                    use_normalized_coordinates=True,
                    line_thickness=8)

                im = Image.fromarray(image_np)
                im.save(Config.OBJ_DETECTION_FOLDER + filename)

    return send_from_directory(Config.OBJ_DETECTION_FOLDER,
                               filename)


# stock price

@app.route('/stock_prediction')
def stock_prediction():
    return render_template('stock_prediction.html')


# face recognition

@app.route('/face_recognize')
def face_recognize():
    return render_template('face_express_recognize.html')


# emotion analysis

@app.route('/emotion_analysis')
def emotion_analysis():
    return render_template('emotion_analysis.html')


# xray
@app.route('/detecting_pneumonia')
def detecting_pneumonia():
    return render_template('chest_xray.html')


# car prices
@app.route('/price_prediction')
def price_prediction():
    return render_template('car_price_predition.html')


if __name__ == "__main__":
    app.run(debug=True)