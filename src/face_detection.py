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


class FaceDetection:
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
        logger.info("Loading the Face Detection Model...")
        self.net = core.load_network(network=model, device_name='CPU', num_requests=1)
        logger.info('Time taken to load the model is: {:.4f} seconds'.format(time.time() - start))

        return

        raise NotImplementedError

    def predict(self, image, perf):
        '''
        TODO: You will need to complete this method.
        This method is meant for running predictions on the input image.
        '''
        pp = pprint.PrettyPrinter(indent=4)
        processed_image = self.preprocess_input(image)
        input_name, input_shape, output_name, output_shape = self.check_model()
        input_dict = {input_name: processed_image}
        start = time.time()
        infer = self.net.start_async(request_id=0, inputs=input_dict)
        self.count += 1
        if self.net.requests[0].wait(-1) == 0:
            results = self.net.requests[0].outputs[output_name]

            if perf == 'yes':
                logger.info('Face Detection Model Inference speed is: {:.3f} fps'.format(1 / (time.time() - start)))
                logger.info('Layers Performance counts:')
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
        image = cv2.resize(image, (input_shape[3], input_shape[2]), interpolation=cv2.INTER_AREA)
        image = image.transpose((2, 0, 1))
        image = image.reshape(1, *image.shape)

        return image

        raise NotImplementedError

    def preprocess_output(self, image, perf):
        '''
        Before feeding the output of this model to the next model,
        you might have to preprocess the output. This function is where you can do that.
        '''
        outputs = self.predict(image, perf)
        h, w, c = image.shape
        for character in (outputs[0][0]):
            if character[2] > 0.6:
                x_min = int(w * character[3])
                y_min = int(h * character[4])
                x_max = int(w * character[5])
                y_max = int(h * character[6])
                crop = image[y_min - 40:y_max + 40, x_min - 50:x_max + 50]
                if perf == 'yes':
                    image = cv2.resize(image, (720, 480), interpolation=cv2.INTER_AREA)
                    image = cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 1)
                    cv2.imshow('Face Detection Results', image)
                return crop

