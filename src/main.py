from src.input_feeder import InputFeeder
from src.face_detection import FaceDetection
from src.head_pose_estimation import HeadPoseEstimation
from src.facial_landmarks_detection import FacialLandmarksDetection
from src.gaze_estimation import GazeEstimation
from src.mouse_controller import MouseController
import argparse
import time
import logging
import cv2

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)


def main(args):
    fd = FaceDetection(args.model_path + 'face-detection-adas-binary-0001/FP32-INT1/face-detection-adas-binary-0001',
                       device=args.device)
    hpe = HeadPoseEstimation(args.model_path + 'head-pose-estimation-adas-0001/FP16-INT8/head-pose-estimation-adas-0001',
                             device=args.device)
    fld = FacialLandmarksDetection(args.model_path + 'landmarks-regression-retail-0009/FP16-INT8/landmarks-regression'
                                                '-retail-0009', device=args.device)
    ge = GazeEstimation(args.model_path + 'gaze-estimation-adas-0002/FP16-INT8/gaze-estimation-adas-0002',
                        device=args.device)

    mc = MouseController('medium', 'fast')

    feed = InputFeeder(input_type=args.input_type, input_file=args.input_file)

    feed.load_data()

    fd.load_model()
    fld.load_model()
    hpe.load_model()
    ge.load_model()

    results_status = args.show_results

    def run_if_no_face_detected():
        try:
            for batch in feed.next_batch():
                start = time.time()
                cropped = fd.preprocess_output(batch, results_status)
                key = cv2.waitKey(1)
                stream = cv2.waitKey(1)
                not_stream = cv2.waitKey(1)
                raw = cv2.resize(batch, (720, 480), interpolation=cv2.INTER_AREA)
                if cropped.shape[2] == 3:
                    right_eye, left_eye = fld.preprocess_output(cropped, results_status)
                    head_angles = hpe.preprocess_output(cropped, results_status)
                    coordinates = ge.predict(right_eye, left_eye, head_angles, results_status)
                    mc.move(coordinates[0][0], coordinates[0][1])
                    if results_status == 'yes':
                        logger.info('The total time taken to obtain results is: {:.4f} seconds'.format(time.time()-start))
                    if key == ord('q'):
                        break
                    if not_stream != ord('t') and (stream == ord('w') or not_stream != ord('t')):
                        cv2.imshow('Raw', raw)
                    if stream != ord('w') and not_stream == ord('t'):
                        cv2.destroyAllWindows()
            feed.close()
        except AttributeError:
            logger.warning('Cannot detected a face on the video frame')
            run_if_no_face_detected()

    run_if_no_face_detected()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_type', required=True, help='Enter the type of input either video, cam or image')
    parser.add_argument('--input_file', default='bin/demo.mp4', help='Enter the directory path for the input file')
    parser.add_argument('--device', default='CPU', help='Enter the name of the device to perform inference on')
    parser.add_argument('--show_results', default='no', help='Enter yes to show and no to hide performance results')
    parser.add_argument('--model_path', required=True, help='Add the path to the directory containing the four models')
    args = parser.parse_args()
    try:
        main(args)
    except KeyboardInterrupt:
        logger.info('Program cancelled by user')