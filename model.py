import os
import numpy as np
import tensorflow as tf
import sys
from projects.object_detection.models.utils import label_map_util
from collections import defaultdict
from config import Config


def init_detection():
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.compat.v1.GraphDef()
        with tf.compat.v2.io.gfile.GFile(Config.OBJ_DETECT_MODEL_PATH, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name="")

    label_map = label_map_util.load_labelmap(Config.OBJ_DETECT_MODEL_LABELS)
    categories = label_map_util.convert_label_map_to_categories(label_map=label_map,max_num_classes=Config.OBJ_DETECT_MODEL_CLASSES, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)

    return detection_graph, category_index

def convert_image_to_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(im_height,im_width,3).astype(np.uint8)







