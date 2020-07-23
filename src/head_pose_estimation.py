import cv2
import logging as log
import os
import numpy as np
from openvino.inference_engine import IENetwork, IECore


class HeadPoseEstimation:
    '''
    Class for the Head Pose Estimation Model.
    '''

    def __init__(self, model_name, device='CPU', extensions=None, threshold=0.6):
        '''
        TODO: Use this to set your instance variables.
        '''
        self.device = device
        self.threshold= threshold
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
        self.exec_network = self.core.load_network(self.network, self.device)
        return self.exec_network

    def predict(self, image):
        '''
        TODO: You will need to complete this method.
        This method is meant for running predictions on the input image.
        '''
        self.preprocess_image = self.preprocess_input(image)
        self.results = self.exec_network.infer(inputs={self.input: self.preprocess_image})
        self.output_list = self.preprocess_output(self.results)

        return self.output_list



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
        image = image.astype(np.float32)
        net_input_shape = self.network.inputs[self.input].shape
        frame = cv2.resize(image, (net_input_shape[3], net_input_shape[2]))
        frame = frame.transpose(2, 0, 1)
        frame = frame.reshape(1, *frame.shape)
        return frame

    def preprocess_output(self, outputs):
        '''
        Before feeding the output of this model to the next model,
        you might have to preprocess the output. This function is where you can do that.
        '''

        output = {
            "angle_y_fc": outputs['angle_y_fc'][0][0],
            "angle_p_fc": outputs['angle_p_fc'][0][0],
            "angle_r_fc": outputs['angle_r_fc'][0][0]
        }

        return output
