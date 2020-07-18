'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''
from openvino.inference_engine import IENetwork, IECore
import cv2
import pprint
import numpy as np
import time
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)


class GazeEstimation:
    '''
    Class for the Face Detection Model.
    '''
    def __init__(self, model_name, device='CPU', extensions=None):
        '''
        TODO: Use this to set your instance variables.
        '''
        self.model_weights = model_name + '.xml'
        self.model_structure = model_name + '.bin'
        self.device = device
        self.net = None

        return

        raise NotImplementedError

    def load_model(self):
        '''
        TODO: You will need to complete this method.
        This method is for loading the model to the device specified by the user.
        If your model requires any Plugins, this is where you can load them.
        '''
        core = IECore()
        start = time.time()
        logger.info('Loading the Gaze Estimation Model...')
        model = core.read_network(self.model_weights, self.model_structure)
        self.net = core.load_network(network=model, device_name='CPU', num_requests=1)
        logger.info('Time taken to load the model is: {:.4f} seconds'.format(time.time()-start))

        return self.net

        raise NotImplementedError

    def predict(self, right_eye, left_eye, head_angles, results):
        '''
        TODO: You will need to complete this method.
        This method is meant for running predictions on the input image.
        '''
        pp = pprint.PrettyPrinter()
        right, left, angles = self.preprocess_input(right_eye, left_eye, head_angles)
        input_dict = {'right_eye_image': right, 'left_eye_image': left, 'head_pose_angles': angles}
        start = time.time()
        infer = self.net.start_async(request_id=0, inputs=input_dict)
        if self.net.requests[0].wait(-1) == 0:
            results = self.net.requests[0].outputs['gaze_vector']
            if results == 'yes':
                logger.info('Gaze Estimation Model Inference speed is: {:.3f} fps'.format(1 / (time.time() - start)))
                logger.info("Gaze Estimation Model Layers performance counts results")
                pp.pprint(infer.get_perf_counts())

        return self.preprocess_output(results)

        raise NotImplementedError

    def check_model(self):
        output_name = next(iter(self.net.outputs))
        output_shape = self.net.outputs[output_name].shape

        return self.net.inputs, output_name, output_shape

        raise NotImplementedError

    def preprocess_input(self, right_eye, left_eye, angles):
        '''
        Before feeding the data into the model for inference,
        you might have to preprocess it. This function is where you can do that.
        '''
        if right_eye.any() and left_eye.any():
            right_eye = cv2.resize(right_eye, (60, 60), interpolation=cv2.INTER_AREA)
            right_eye = right_eye.transpose((2, 0, 1))
            right_eye = right_eye.reshape(1, *right_eye.shape)

            left_eye = cv2.resize(left_eye, (60, 60), interpolation=cv2.INTER_AREA)
            left_eye = left_eye.transpose((2, 0, 1))
            left_eye = left_eye.reshape(1, *left_eye.shape)

            angles = np.array([angles[0], angles[1], angles[2]])
            angles = angles.reshape(1, 3)

        return right_eye, left_eye, angles

        raise NotImplementedError

    def preprocess_output(self, outputs):
        '''
        Before feeding the output of this model to the next model,
        you might have to preprocess the output. This function is where you can do that.
        '''

        return outputs

        raise NotImplementedError