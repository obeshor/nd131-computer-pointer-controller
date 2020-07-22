'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''
from openvino.inference_engine import IENetwork, IECore
import cv2
import numpy as np
import time
import logging
import sys


class FaceDetection:
    '''
    Class for the Face Detection Model.
    '''

    def __init__(self, model_name, device='CPU', extensions=None):

        self.model_name = model_name
        self.device = device
        self.extensions = extensions
        self.plugin = None
        self.network = None
        self.exec_net = None
        self.in_name = None
        self.in_shape = None
        self.out_name = None
        self.out_shape = None

    def load_model(self):

        '''
        This method is for loading the model to the device specified by the user.
        If your model requires any Plugins, this is where you can load them.
        '''

        model_structure = self.model_name
        model_weights = self.model_name.split('.')[0] + '.bin'

        self.plugin = IECore()

        if self.extensions and 'CPU' in self.device:
            self.plugin.add_extension(self.extensions, self.device)

        self.network = IENetwork(model=model_structure, weights=model_weights)

        self.check_model()

        self.exec_net = self.plugin.load_network(network=self.network, device_name=self.device, num_requests=1)

        self.in_name = next(iter(self.network.inputs))
        self.in_shape = self.network.inputs[self.in_name].shape

        self.out_name = next(iter(self.network.outputs))
        self.out_shape = self.network.outputs[self.out_name].shape

    def predict(self, image, prob_threshold):

        processed_image = self.preprocess_input(image.copy())
        outputs = self.exec_net.infer({self.in_name: processed_image})
        coord = self.preprocess_output(outputs, prob_threshold)

        if (len(coord) == 0):
            return 0, 0

        coord = coord[0]

        height = image.shape[0]
        width = image.shape[1]

        coord = coord * np.array([width, height, width, height])
        coord = coord.astype(np.int32)

        face = image[coord[1]:coord[3], coord[0]:coord[2]]
        return face, coord

    def check_model(self):

        if self.device == "CPU":
            supported_layers = self.plugin.query_network(network=self.network, device_name=self.device)
            notsupported_layers = [l for l in self.network.layers.keys() if l not in supported_layers]

            if len(notsupported_layers) != 0:
                logging.error("[ERROR] Unsupported layers found: {}".format(notsupported_layers))
                sys.exit(1)

    def preprocess_input(self, image):

        image_processed = cv2.resize(image, (self.in_shape[3], self.in_shape[2]))
        image_processed = image_processed.transpose(2, 0, 1)
        image_processed = image_processed.reshape(1, *image_processed.shape)
        return image_processed

    def preprocess_output(self, outputs, prob_threshold):

        coord = []
        outs = outputs[self.out_name][0][0]
        for out in outs:
            conf = out[2]
            if conf > prob_threshold:
                x_min = out[3]
                y_min = out[4]
                x_max = out[5]
                y_max = out[6]
                coord.append([x_min, y_min, x_max, y_max])
        return coord

