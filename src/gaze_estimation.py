'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''

import os
import cv2
import math
import numpy as np
from openvino.inference_engine import IECore

class GazeEstimation:
    '''
    Class for the Face Detection Model.
    '''
    def __init__(self, model_name, device='CPU', extensions=None, threshold = 0.6):
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

        self.input_shape = self.network.inputs[self.input].shape

    def load_model(self):
        '''
        TODO: You will need to complete this method.
        This method is for loading the model to the device specified by the user.
        If your model requires any Plugins, this is where you can load them.
        '''
        self.exec_network = self.core.load_network(self.network, self.device)
        return self.exec_network

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
            n,c = self.input_shape
            image = cv2.resize(image, (60,60))
            image = image.transpose((2,0,1))
            image = image.reshape(n,c,60,60)
        except Exception as e:
            print("Error While preprocessing Image in " + str(e))
        return image

    def predict(self, left_eye, right_eye, head_pose_angles, cropped_face, eyes_coords):
        """
        Make inference over the exectutable network
        """
        left_eye_image = self.preprocess_input(left_eye)
        right_eye_image = self.preprocess_input(right_eye)

        outputs = self.exec_network.infer({"head_pose_angles": head_pose_angles,
                                           "left_eye_image": left_eye_image,
                                           "right_eye_image": right_eye_image
                                           })

        x = round(outputs[self.output][0][0], 4)
        y = round(outputs[self.output][0][1], 4)
        z = outputs[self.output][0][2]

        center_x_left_eye = int((eyes_coords[0][1][0] - eyes_coords[0][0][0]) / 2 + eyes_coords[0][0][0])
        center_y_left_eye = int((eyes_coords[0][1][1] - eyes_coords[0][0][1]) / 2 + eyes_coords[0][0][1])
        new_x_left_eye = int(center_x_left_eye + x * 40)
        new_y_left_eye = int(center_y_left_eye + y * 40 * -1)
        cv2.line(cropped_face, (center_x_left_eye, center_y_left_eye), (new_x_left_eye, new_y_left_eye), (0, 255, 0), 2)

        center_x_right_eye = int((eyes_coords[1][1][0] - eyes_coords[1][0][0]) / 2 + eyes_coords[1][0][0])
        center_y_right_eye = int((eyes_coords[1][1][1] - eyes_coords[1][0][1]) / 2 + eyes_coords[1][0][1])
        new_x_right_eye = int(center_x_right_eye + x * 40)
        new_y_right_eye = int(center_y_right_eye + y * 40 * -1)
        cv2.line(cropped_face, (center_x_right_eye, center_y_right_eye), (new_x_right_eye, new_y_right_eye),
                 (0, 255, 0), 2)

        return x, y, z


    def preprocess_output(self, outputs, head_position):
        '''
        Before feeding the output of this model to the next model,
        you might have to preprocess the output. This function is where you can do that.
        '''
        roll = head_position[2]
        gaze_vector = outputs / cv2.norm(outputs)

        cosValue = math.cos(roll * math.pi / 180.0)
        sinValue = math.sin(roll * math.pi / 180.0)

        x = gaze_vector[0] * cosValue * gaze_vector[1] * sinValue
        y = gaze_vector[0] * sinValue * gaze_vector[1] * cosValue
        return (x, y)

