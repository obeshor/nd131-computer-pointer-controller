from argparse import ArgumentParser
from input_feeder import InputFeeder
import os
from face_detection import FaceDetection
from facial_landmarks_detection import FacialLandmarksDetection
from gaze_estimation import GazeEstimation
from head_pose_estimation import HeadPoseEstimation
from mouse_controller import MouseController
import cv2
import logging as log
import numpy as np
import time


def build_argparser():
    """
    parse commandline argument
    return ArgumentParser object
    """
    parser = ArgumentParser()
    parser.add_argument("-fd", "--faceDetectionModel", type=str, required=True,
                        help="Path to a face detection model xml file with a trained model.")

    parser.add_argument("-fl", "--facialLandmarksModel", type=str, required=True,
                        help="Path to a facial landmarks detection model xml file with a trained model.")

    parser.add_argument("-hp", "--headPoseEstimationModel", type=str, required=True,
                        help="Path to a head pose estimation model xml file with a trained model")

    parser.add_argument("-ge", "--gazeEstimationModel", type=str, required=True,
                        help="Path to a gaze estimation model xml file with a trained model.")

    parser.add_argument("-i", "--input", type=str, required=True,
                        help="Path to image or video file or CAM")

    parser.add_argument("-flags", "--previewFlags", required=False, nargs='+',
                        default=[],
                        help="Specify flag from fd, fl, hp, ge like -flags fd fl(Space separated if multiple values)"
                             "fd for faceDetectionModel, fl for facialLandmarkModel"
                             "hp for headPoseEstimationModel, ge for gazeEstimationModel")

    parser.add_argument("-prob", "--prob_threshold", required=False, type=float,
                        default=0.6,
                        help="Probability threshold for detections filtering"
                             "(0.6 by default)")

    parser.add_argument("-d", "--device", required=False, type=str, default='CPU',
                        help="Specify the target device to infer on"
                             "It can be CPU(Default), GPU, FPGU, MYRIAD")
    parser.add_argument("-o", '--output_path', default='/results/', type=str)
    return parser

def main():
    args = build_argparser().parse_args()
    device_name = args.device
    prob_threshold = args.prob_threshold
    logger_object = log.getLogger()

    # Initialize variables with the input arguments
    model_path_dict = {
        'FaceDetectionModel': args.faceDetectionModel,
        'FacialLandmarkModel': args.facialLandmarksModel,
        'HeadPoseEstimationModel': args.headPoseEstimationModel,
        'GazeEstimationModel': args.gazeEstimationModel
    }

    # Instantiate model
    face_model = FaceDetection(model_path_dict['FaceDetectionModel'], device_name, threshold=prob_threshold)
    landmark_model = FacialLandmarksDetection(model_path_dict['FacialLandmarkModel'], device_name,
                                              threshold=prob_threshold)
    head_pose_model = HeadPoseEstimation(model_path_dict['HeadPoseEstimationModel'], device_name,
                                         threshold=prob_threshold)
    gaze_model = GazeEstimation(model_path_dict['GazeEstimationModel'], device_name, threshold=prob_threshold)
    mouse_controller = MouseController('medium', 'fast')

    # Load Models and get time
    start_time = time.time()
    face_model.load_model()
    logger_object.error("Face detection model loaded: time: {:.3f} ms".format((time.time() - start_time) * 1000))

    first_mark = time.time()
    landmark_model.load_model()
    logger_object.error(
        "Facial landmarks detection model loaded: time: {:.3f} ms".format((time.time() - first_mark) * 1000))

    second_mark = time.time()
    head_pose_model.load_model()
    logger_object.error("Head pose estimation model loaded: time: {:.3f} ms".format((time.time() - second_mark) * 1000))

    third_mark = time.time()
    gaze_model.load_model()
    logger_object.error("Gaze estimation model loaded: time: {:.3f} ms".format((time.time() - third_mark) * 1000))
    load_total_time = time.time() - start_time
    logger_object.error("Total loading time: time: {:.3f} ms".format(load_total_time * 1000))
    logger_object.error("All models are loaded successfully..")

    # Check extention of these unsupported layers
    face_model.check_model()
    landmark_model.check_model()
    head_pose_model.check_model()
    gaze_model.check_model()

    preview_flags = args.previewFlags
    input_filename = args.input
    output_path = args.output_path
    prob_threshold = args.prob_threshold

    if input_filename.lower() == 'cam':
        input_feeder = InputFeeder(input_type='cam')
    else:
        if not os.path.isfile(input_filename):
            logger_object.error("Unable to find specified video file")
            exit(1)
        input_feeder = InputFeeder(input_type='video', input_file=input_filename)

    for model_path in list(model_path_dict.values()):
        if not os.path.isfile(model_path):
            logger_object.error("Unable to find specified model file" + str(model_path))
            exit(1)

    input_feeder.load_data()
    width = int(input_feeder.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(input_feeder.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(input_feeder.cap.get(cv2.CAP_PROP_FPS))
    out_video = cv2.VideoWriter(os.path.join('output_video.mp4'), cv2.VideoWriter_fourcc(*'avc1'), fps,
                                (width, height), True)

    frame_counter = 0
    start_inf_time = time.time()
    for ret, frame in input_feeder.next_batch():
        if not ret:
            break
        frame_counter += 1
        key = cv2.waitKey(60)

        try:
            cropped_image, face_cords = face_model.predict(frame, prob_threshold)

            if type(cropped_image) == int:
                print("Unable to detect the face")
                if key == 27:
                    break
                continue

            left_eye, right_eye, eye_cords = landmark_model.predict(cropped_image)
            pose_output = head_pose_model.predict(cropped_image)
            x, y, z = gaze_model.predict(left_eye, right_eye, pose_output, cropped_image, eye_cords)

            mouse_controller.move(x, y)
        except Exception as e:
            print(str(e) + " for frame " + str(frame_counter))
            continue

        image = cv2.resize(frame, (width, height))
        if not len(preview_flags) == 0:
            preview_frame = frame.copy()

            if 'fd' in preview_flags:
                if len(preview_flags) != 1:
                    preview_frame = cropped_image
                    cv2.rectangle(frame, (face_cords[0], face_cords[1]), (face_cords[2], face_cords[3]), (0, 0, 255), 3)

            if 'hp' in preview_flags:
                cv2.putText(
                    frame,
                    "Pose Angles: yaw= {:.2f} , pitch= {:.2f} , roll= {:.2f}".format(
                        pose_output[0], pose_output[1], pose_output[2]),
                    (20, 40),
                    cv2.FONT_HERSHEY_DUPLEX,
                    1, (255, 0, 0), 3)

            if 'ge' in preview_flags:
                cv2.putText(
                    frame,
                    "Gaze vector: x= {:.2f} , y= {:.2f} , z= {:.2f}".format(
                        x, y, z),
                    (15, 100),
                    cv2.FONT_HERSHEY_COMPLEX,
                    1, (0, 255, 0), 3)

            image = np.hstack((cv2.resize(frame, (500, 500)), cv2.resize(preview_frame, (500, 500))))

        cv2.imshow('preview', image)
        out_video.write(image)

        if frame_counter % 5 == 0:
            mouse_controller.move(x, y)

        if key == 27:
            break

    inference_time = round(time.time() - start_inf_time, 1)
    fps = int(frame_counter) / inference_time
    logger_object.error("counter {} seconds".format(frame_counter))
    logger_object.error("total inference time {} seconds".format(inference_time))
    logger_object.error("fps {} frame/second".format(fps))
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'stats.txt'), 'w') as f:
        f.write('inference time : ' + str(inference_time) + '\n')
        f.write('fps: ' + str(fps) + '\n')
        f.write('Models Loading: '+ str(load_total_time) + '\n')
    logger_object.error('Video stream ended')
    cv2.destroyAllWindows()
    input_feeder.close()


if __name__ == '__main__':
    main()