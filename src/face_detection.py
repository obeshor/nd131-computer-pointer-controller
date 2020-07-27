'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''
import os
import cv2
import numpy as np
from openvino.inference_engine import IECore


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
        self.network = self.core.read_network(model=str(model_name), weights=str(os.path.splitext(model_name)[0] + ".bin"))
        self.input = next(iter(self.network.inputs))
        self.output = next(iter(self.network.outputs))

    def load_model(self):
        '''
        TODO: You will need to complete this method.
        This method is for loading the model to the device specified by the user.
        If your model requires any Plugins, this is where you can load them.
        '''
        self.exec_network = self.core.load_network(self.network, self.device)
        return self.exec_network

    def predict(self, image, prob_threshold):
        '''
        TODO: You will need to complete this method.
        This method is meant for running predictions on the input image.
        '''
        img_processed = self.preprocess_input(image.copy())
        outputs = self.exec_network.infer({self.input: img_processed})
        faces_coordinates = self.preprocess_output(outputs, prob_threshold)
        if (len(faces_coordinates) == 0):
            return 0, 0
        faces_coordinates = faces_coordinates[0]
        height = image.shape[0]
        width = image.shape[1]
        faces_coordinates = faces_coordinates * np.array([width, height, width, height])
        faces_coordinates = faces_coordinates.astype(np.int32)
        cropped_face_image = image[faces_coordinates[1]:faces_coordinates[3], faces_coordinates[0]:faces_coordinates[2]]
        return cropped_face_image, faces_coordinates

    def check_model(self):
        supported_layers = self.core.query_network(network=self.network, device_name=self.device)
        unsupported_layers = [layer for layer in self.network.layers.keys() if layer not in supported_layers]
        if len(unsupported_layers) > 0:
            print("Check extention of these unsupported layers =>" + str(unsupported_layers))
            exit(1)
        print("All layers are supported")

    def preprocess_input(self, image):
        '''
        Before feeding the data into the model for inference,
        you might have to preprocess it. This function is where you can do that.
        '''
        net_input_shape = self.network.inputs[self.input].shape
        p_frame = cv2.resize(image, (net_input_shape[3], net_input_shape[2]))
        p_frame = p_frame.transpose(2, 0, 1)
        p_frame = p_frame.reshape(1, *p_frame.shape)
        return p_frame

    def preprocess_output(self, outputs, prob_threshold):
        '''
        Before feeding the output of this model to the next model,
        you might have to preprocess the output. This function is where you can do that.
        '''
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