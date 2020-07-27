'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''
import os
import cv2
import numpy as np
from openvino.inference_engine import IENetwork, IECore


class FacialLandmarksDetection:
    '''
    Class for the Face Detection Model.
    '''

    def __init__(self, model_name, device='CPU', extensions=None, threshold=0.6):
        '''
        TODO: Use this to set your instance variables.
        '''
        self.device = device
        self.threshold = threshold

        model_bin = os.path.splitext(model_name)[0] + ".bin"
        try:
            self.network = IENetwork(model_name, model_bin)
        except Exception as e:
            print("Cannot initialize the network. Please enter correct model path. Error : %s", e)

        self.core = IECore()
        self.input_name = next(iter(self.network.inputs))
        self.output_name = next(iter(self.network.outputs))
        self.input_shape = self.network.inputs[self.input_name].shape
        self.width = None
        self.height = None

    def load_model(self):
        '''
        TODO: You will need to complete this method.
        This method is for loading the model to the device specified by the user.
        If your model requires any Plugins, this is where you can load them.
        '''
        self.exec_network = self.core.load_network(self.network, self.device)
        return self.exec_network

    def predict(self, image):
        """
        Make inference over the exectutable network
        """
        self.width = image.shape[1]
        self.height = image.shape[0]
        p_frame = self.preprocess_input(image)

        outputs = self.exec_network.infer({self.input_name: p_frame})

        left_eye, right_eye = self.preprocess_outputs(outputs[self.output_name])

        # cropped image for left eye
        y_left_eye = int(left_eye[1])
        x_left_eye = int(left_eye[0])
        cropped_left_eye = image[(y_left_eye - 20):(y_left_eye + 20), (x_left_eye - 20):(x_left_eye + 20)]

        # cropped image for Right eye
        y_right_eye = int(right_eye[1])
        x_right_eye = int(right_eye[0])
        cropped_right_eye = image[(y_right_eye - 20):(y_right_eye + 20), (x_right_eye - 20):(x_right_eye + 20)]

        # eye coords
        eyes_coords = [[(x_left_eye - 20, y_left_eye - 20), (x_left_eye + 20, y_left_eye + 20)],
                       [(x_right_eye - 20, y_right_eye - 20), (x_right_eye + 20, y_right_eye + 20)]]

        cv2.rectangle(image, (eyes_coords[0][0][0], eyes_coords[0][0][1]),
                      (eyes_coords[0][1][0], eyes_coords[0][1][1]), (255, 0, 0), 2)
        cv2.rectangle(image, (eyes_coords[1][0][0], eyes_coords[1][0][1]),
                      (eyes_coords[1][1][0], eyes_coords[1][1][1]), (255, 0, 0), 2)

        return cropped_left_eye, cropped_right_eye, eyes_coords

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
        try:
            image = image.astype(np.float32)
            n, c, h, w = self.input_shape
            image = cv2.resize(image, (w, h))
            image = image.transpose((2, 0, 1))
            image = image.reshape(n, c, h, w)
        except Exception as e:
            print("Error While preprocessing Image in " + str(e))
        return image

    def preprocess_outputs(self, outputs):
        """
        The return will contain the related coordinates of the prediction, resized to the original image size
        """
        left_eye = (outputs[0][0][0][0] * self.width, outputs[0][1][0][0] * self.height)
        right_eye = (outputs[0][2][0][0] * self.width, outputs[0][3][0][0] * self.height)
        return left_eye, right_eye