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


class FacialLandmarksDetection:
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
        self.count = 0

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
        model = core.read_network(self.model_weights, self.model_structure)
        logger.info('Loading the Facial Landmarks Detection Model...')
        self.net = core.load_network(network=model, device_name='CPU', num_requests=1)
        logger.info('Time taken to load the model is: {:.4f} seconds'.format(time.time()-start))

        return self.net

        raise NotImplementedError

    def predict(self, image, results):
        '''
        TODO: You will need to complete this method.
        This method is meant for running predictions on the input image.
        '''
        pp = pprint.PrettyPrinter()
        processed_image = self.preprocess_input(image)
        input_name, input_shape, output_name, output_shape = self.check_model()
        input_dict = {input_name: processed_image}
        infer = self.net.start_async(request_id=0, inputs=input_dict)
        self.count += 1
        start = time.time()
        if self.net.requests[0].wait(-1) == 0:
            results = self.net.requests[0].outputs[output_name]

            if results == 'yes':
                logger.info('Facial Landmarks Detection Inference speed is: {:.3f} fps'.format(1 / (time.time() - start)))
                logger.info('Facial Landmarks Detection Model performance counts results')
                pp.pprint(infer.get_perf_counts())

        return results

        raise NotImplementedError

    def check_model(self):
        input_name = next(iter(self.net.inputs))
        input_shape = self.net.inputs[input_name].shape
        output_name = next(iter(self.net.outputs))
        output_shape = self.net.outputs[output_name].shape

        return input_name, input_shape, output_name, output_shape

        raise NotImplementedError

    def preprocess_input(self, image):
        '''
        Before feeding the data into the model for inference,
        you might have to preprocess it. This function is where you can do that.
        '''
        input_name, input_shape, output_name, output_shape = self.check_model()
        if image.any():
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
        outputs = self.predict(image, results)
        h, w, c = image.shape
        x0, y0 = int(w*outputs[0][0][0][0]), int(h*outputs[0][1][0][0])
        x1, y1 = int(w*outputs[0][2][0][0]), int(h*outputs[0][3][0][0])
        if results == 'yes':
            image = cv2.rectangle(image, (x0 - 20, y0 - 20), (x0 + 20, y0 + 20), (0, 255, 0), 1)
            image = cv2.rectangle(image, (x1 - 20, y1 - 20), (x1 + 20, y1 + 20), (0, 255, 0), 1)
            cv2.imshow('Facial Landmarks', image)
        right_eye = image[y0-30: y0+30, x0-30:x0+30]
        left_eye = image[y1 - 30: y1 + 30, x1 - 30:x1 + 30]

        return right_eye, left_eye

        raise NotImplementedError