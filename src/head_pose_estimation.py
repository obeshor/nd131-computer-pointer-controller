'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''
from openvino.inference_engine import IENetwork, IECore
import pprint
import cv2
import time
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)


class HeadPoseEstimation:
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
        logger.info("Loading the Head Pose Estimation Model...")
        model = core.read_network(self.model_weights, self.model_structure)
        self.net = core.load_network(network=model, device_name='CPU', num_requests=1)
        logger.info('Time taken to load the model is: {:.4f} seconds'.format(time.time() - start))

        return self.net

        raise NotImplementedError

    def predict(self, image, results):
        '''
        TODO: You will need to complete this method.
        This method is meant for running predictions on the input image.
        '''
        pp = pprint.PrettyPrinter()
        processed_image = self.preprocess_input(image)
        input_name, input_shape, output_name_one, output_name_two, output_name_three, output_shape_one, \
        output_shape_two, output_shape_three = self.check_model()
        input_dict = {input_name: processed_image}
        start = time.time()
        infer = self.net.start_async(request_id=0, inputs=input_dict)
        if self.net.requests[0].wait(-1) == 0:
            results_one = self.net.requests[0].outputs[output_name_one]
            results_two = self.net.requests[0].outputs[output_name_two]
            results_three = self.net.requests[0].outputs[output_name_three]
            if results == 'yes':
                logger.info('Head Pose Estimation Model Inference speed is: {:.3f} fps'.format(1 / (time.time() - start)))
                logger.info('Head Pose Estimation Model Layers performance counts:')
                pp.pprint(infer.get_perf_counts())

        return results_one, results_two, results_three

        raise NotImplementedError

    def check_model(self):
        input_name = next(iter(self.net.inputs))
        input_shape = self.net.inputs[input_name].shape
        output_name_one, output_name_two, output_name_three = self.net.outputs
        output_shape_one = self.net.outputs[output_name_one].shape
        output_shape_two = self.net.outputs[output_name_two].shape
        output_shape_three = self.net.outputs[output_name_three].shape

        return input_name, input_shape, output_name_one, output_name_two, output_name_three, \
               output_shape_one, output_shape_two, output_shape_three

        raise NotImplementedError

    def preprocess_input(self, image):
        '''
        Before feeding the data into the model for inference,
        you might have to preprocess it. This function is where you can do that.
        '''
        input_name, input_shape, output_name_one, output_name_two, output_name_three, \
        output_shape_one, output_shape_two, output_shape_three= self.check_model()
        image = cv2.resize(image, (input_shape[3], input_shape[2]), interpolation=cv2.INTER_AREA)
        image = image.transpose((2, 0, 1))
        image = image.reshape(1, *image.shape)

        return image

        raise NotImplementedError

    def preprocess_output(self, image, results):
        '''
        Before feeding the output of this model to the next model,
        you might have to preprocess the output. This function is where you can do that.
        '''
        yaw, pitch, roll = self.predict(image, results)
        angles = [yaw[0], pitch[0], roll[0]]

        return angles

        raise NotImplementedError