import cv2
import logging as log
import os
import numpy as np
from openvino.inference_engine import IENetwork, IECore
import math


class GazeEstimation:
    '''
    Class for the Gaze Estimation Model.
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
        self.input_shape = self.network.inputs[self.input].shape

    def load_model(self):
        '''
        TODO: You will need to complete this method.
        This method is for loading the model to the device specified by the user.
        If your model requires any Plugins, this is where you can load them.
        '''
        self.exec_network = self.core.load_network(self.network, self.device)
        return self.exec_network

    def predict(self, left_eye_image, right_eye_image, head_pose_output):
        '''
        This method is meant for running predictions on the input image.
        '''
        left_eye_image = self.preprocess_input(left_eye_image)
        right_eye_image = self.preprocess_input(right_eye_image)
        input_dict = {
            "left_eye_image": left_eye_image,
            "right_eye_image": right_eye_image,
            "head_pose_output": head_pose_output
        }

        self.exec_network.start_async(request_id=0, inputs=input_dict)
        status = self.exec_network.requests[0].wait(-1)
        if status == 0:
            result = self.exec_network.requests[0].outputs

        coords = self.preprocess_output(result)

        return coords

    def check_model(self):

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
        you might have to preprocess it. This function is where you can do that.
        '''
        try:
            image = image.astype(np.float32)
            n, c = self.input_shape
            image = cv2.resize(image, (60, 60))
            image = image.transpose((2, 0, 1))
            image = image.reshape(n, c, 60, 60)
        except Exception as e:
            print("Error While preprocessing Image in " + str(e))
        return image

    def preprocess_output(self, outputs, head_pose_estimation_output):
        '''
        Before feeding the output of this model to the next model,
        you might have to preprocess the output. This function is where you can do that.
        '''
        roll = head_pose_estimation_output[2]
        gaze_vector = outputs / cv2.norm(outputs)

        cosValue = math.cos(roll * math.pi / 180.0)
        sinValue = math.sin(roll * math.pi / 180.0)

        x = gaze_vector[0] * cosValue * gaze_vector[1] * sinValue
        y = gaze_vector[0] * sinValue * gaze_vector[1] * cosValue

        return (x, y)
