'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''
import cv2
import logging as log
import os
import numpy as np
from openvino.inference_engine import IENetwork, IECore


class FaceDetection:
    '''
    Class for the Face Detection Model.
    '''

    def __init__(self, model_name, device='CPU', extensions=None, threshold=0.6):
        '''
        TODO: Use this to set your instance variables.
        '''
        self.device = device
        self.threshold = threshold
        self.core = IECore()
        self.network = self.core.read_network(model=str(model_name),
                                              weights=str(os.path.splitext(model_name)[0] + ".bin"))

        self.input = next(iter(self.network.inputs))
        self.output = next(iter(self.network.outputs))

    def load_model(self):
        '''
        This method is for loading the model to the device specified by the user.
        If your model requires any Plugins, this is where you can load them.
        '''

        try:
            self.exec_network = self.core.load_network(self.network, self.device)

        except Exception as e:
            raise ValueError("Could not Initialise the network. Have you enterred the correct model path?")

        return self.exec_network

    def predict(self, image, prob_threshold):
        '''
        This method is meant for running predictions on the input image.
        '''

        img_processed = self.preprocess_input(image.copy())
        outputs = self.exec_network.infer({self.input: img_processed})
        faces_coordinates = self.preprocess_output(outputs, prob_threshold)

        if (len(faces_coordinates) == 0):

            log.error("No Face is detected, Next frame will be processed..")
            return 0, 0

        faces_coordinates = faces_coordinates[0]
        h = image.shape[0]
        w = image.shape[1]
        faces_coordinates = faces_coordinates * np.array([w, h, w, h])
        faces_coordinates = faces_coordinates.astype(np.int32)
        cropped_face_image = image[faces_coordinates[1]:faces_coordinates[3], faces_coordinates[0]:faces_coordinates[2]]

        return cropped_face_image, faces_coordinates

    def check_model(self):
        '''
        Checking for unsupported layers
        '''
        supported_layers = self.core.query_network(network=self.network, device_name=self.device)
        unsupported_layers = [layer for layer in self.network.layers.keys() if layer not in supported_layers]

        if len(unsupported_layers) != 0:
            log.error("Unsupported layers found ...")
            log.error("Adding specified extension")
            self.core.add_extension(self.extension, self.device)
            supported_layers = self.core.query_network(network=self.network, device_name=self.device)
            unsupported_layers = [R for R in self.network.layers.keys() if R not in supported_layers]
            if len(unsupported_layers) != 0:
                log.error("ERROR: There are still unsupported layers after adding extension...")
                exit(1)
        log.info("All Layers supported")

    def preprocess_input(self, image):
        '''
        Before feeding the data into the model for inference,
        you might have to preprocess it. This method is where you can do that.
        '''
        frame = cv2.resize(image, (self.input_shape[3], self.input_shape[2]))
        frame = frame.transpose((2, 0, 1))
        frame = frame.reshape(1, *frame.shape)

        return frame

    def preprocess_output(self, outputs, prob_threshold):

        faces_coordinates = []
        output = outputs[self.output][0][0]
        for box in output:
            conf = box[2]
            if conf >= prob_threshold:
                xmin = box[3]
                ymin = box[4]
                xmax = box[5]
                ymax = box[6]
                faces_coordinates.append([xmin, ymin, xmax, ymax])

        return faces_coordinates
