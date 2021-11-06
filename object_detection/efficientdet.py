import json

import cv2
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

class EfficientNet(object):
    def __init__(self, frozen_graph_path, label_map_path, config_path):
        """Initialize the inference driver
        Args:
          config_path: path to the config.json file
          model_path: path to the frozen inference graph .pb file
          label_map_path: path to the label_map.json file
          threshold: minimal score threshold for filtering predictions
          filter_classes: list of classes to be filtered, remaining classes will be ignored
          model_type: type of the model - "efficientdet" in this case
        """

        self.config_path = config_path
        self.model_path = frozen_graph_path
        self.label_map_path = label_map_path
        
        self.signatures = None
        self.sess = None
        self.load()

        with open(self.label_map_path) as json_file:
            self.idx2label = json.load(json_file, object_hook=lambda d: {int(k): v for k, v in d.items()})
        
        with open(self.config_path ) as json_file:
            data = json.load(json_file)
            self.threshold = data.get('min_conf_threshold', 0.)
            self.filter_classes = data.get('filter_classes', [])
            self.model_type = data.get('model_type', '')

        self.signatures = {
            'image_files': 'image_files:0',
            'image_arrays': 'image_arrays:0',
            'prediction': 'detections:0',
        }

    def _build_session(self):
        sess_config = tf.compat.v1.ConfigProto()
        return tf.compat.v1.Session(config=sess_config)

    def load(self):
        """Load the model using a frozen graph"""

        if not self.sess:
            self.sess = self._build_session()

        graph_def = tf.GraphDef()
        with tf.gfile.GFile(self.model_path, 'rb') as f:
            graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name='')

    def predict(self, image_arrays):
        """Serve a list of image arrays.
        Args:
          image_arrays: A list of image content with each image has shape [height, width, 3] and uint8 type
        Returns:
          A list of detections
        """

        predictions = self.sess.run(
            self.signatures['prediction'],
            feed_dict={self.signatures['image_arrays']: image_arrays})
        filtered_detections = self.format_predictions(
            predictions[0],
            self.threshold,
            self.filter_classes)

        return filtered_detections

    def format_predictions(self, predictions, threshold, filter_classes):
        
        detections = []
        scores=[]

        for id, detection in enumerate(predictions):

            # each detection has a format [image_id, y, x, height, width, score, class]
            _, y, x, h, w, score, class_id = detection
            obj_class = self.idx2label.get(int(class_id), 'other')

            if score >= threshold:
                if (len(filter_classes) >= 1 and obj_class in filter_classes) or len(filter_classes) == 0:
                    scores.append(score)
                    detections.append([int(x), int(y), int(w-x), int(h-y)])

        return detections, scores
