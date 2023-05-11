"""
OpenVINO face recognition module for uh... recognizing faces.
"""

import os
import time
import numpy as np
import cv2
from openvino.inference_engine import IECore
from helpers import utilities
from helpers.utilities import Extras, LOGGER

FACE_TARGET_DEVICE = 'MYRIAD'
FACE_MODEL_XML = 'helpers/openvino/models/intel/face-detection-retail-0005/FP32/face-detection-retail-0005.xml'
FACE_MODEL_BIN = 'helpers/openvino/models/intel/face-detection-retail-0005/FP32/face-detection-retail-0005.bin'
FACE_DETECTION_THRESHOLD = 0.5

AG_TARGET_DEVICE = 'MYRIAD'
AG_MODEL_XML = 'helpers/openvino/models/intel/age-gender-recognition-retail-0013/FP32/age-gender-recognition-retail-0013.xml'
AG_MODEL_BIN = 'helpers/openvino/models/intel/age-gender-recognition-retail-0013/FP32/age-gender-recognition-retail-0013.bin'
AG_MAX_BATCH_SIZE = 1
AG_DYNAMIC_BATCH = False


def crop_frame(frame, coordinate, normalized=True):
    """
    Crop Frame as Given Coordinates.
    """

    x1 = coordinate[0]
    y1 = coordinate[1]
    x2 = coordinate[2]
    y2 = coordinate[3]

    if normalized:
        h = frame.shape[0]
        w = frame.shape[1]

        x1 = int(x1 * w)
        x2 = int(x2 * w)

        y1 = int(y1 * h)
        y2 = int(y2 * h)

    return frame[y1:y2, x1:x2]


def run_app():
    """
    Do stuff! Probably gonna delete this later though.
    """

    openvino_ie = IECore()

    LOGGER.info(f'Available Devices: {openvino_ie.available_devices}')

    if FACE_TARGET_DEVICE == 'CPU' or AG_TARGET_DEVICE == 'CPU':
        openvino_ie.add_extension(
            '/opt/intel/openvino/inference_engine/lib/intel64/libcpu_extension.so', 'CPU')

    LOGGER.info('Loading Face Detection Model....')

    face_detection_network = openvino_ie.read_network(
        model=FACE_MODEL_XML, weights=FACE_MODEL_BIN)
    face_detection_input_layer = next(iter(face_detection_network.inputs))
    face_detection_output_layer = next(iter(face_detection_network.outputs))
    face_detection_input_shape = face_detection_network.inputs[face_detection_input_layer].shape
    face_detection_output_shape = face_detection_network.outputs[
        face_detection_output_layer].shape
    face_detection_executable = openvino_ie.load_network(
        network=face_detection_network, device_name=FACE_TARGET_DEVICE)

    LOGGER.info('Loading Age - Gender Detection Model.....')

    age_gender_detection_network = openvino_ie.read_network(
        model=AG_MODEL_XML, weights=AG_MODEL_BIN)
    age_gender_detection_input_layer = next(
        iter(age_gender_detection_network.inputs))
    age_gender_detection_output_layers = list(
        age_gender_detection_network.outputs.keys())
    age_gender_detection_input_shape = age_gender_detection_network.inputs[
        age_gender_detection_input_layer].shape
    age_output_shape = age_gender_detection_network.outputs[
        age_gender_detection_output_layers[0]].shape
    gender_output_shape = age_gender_detection_network.outputs[
        age_gender_detection_output_layers[0]].shape
    age_gender_detection_network.batch_size = int(AG_MAX_BATCH_SIZE)

    # Check if Dynamic Batching Enabled for Age Gender Detection
    config = {}

    # Get the Batch Size and Allocate Input for Dynamic Batch Process
    NAG, CAG, HAG, WAG = age_gender_detection_network.inputs[age_gender_detection_input_layer].shape

    if AG_DYNAMIC_BATCH:
        config = {'DYN_BATCH_ENABLED': 'YES'}
        LOGGER.info('Dynamic Batch Enabled')

    if NAG > 1:
        age_detection_input = np.zeros(shape=(NAG, CAG, HAG, WAG), dtype=float)

    # Load Executable Network
    age_gender_detection_executable = openvino_ie.load_network(
        network=age_gender_detection_network, config=config, device_name=AG_TARGET_DEVICE)

    # Get Shape Values for Face Detection Network
    N, C, H, W = face_detection_network.inputs[face_detection_input_layer].shape

    # Generate a Named Window
    cv2.namedWindow('Window', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Window', 800, 600)

    cam_data = Extras.CameraData()
    cam_data.load_cam_data(os.getenv('FOSCAM_DATA_FILE'))
    cam_config = cam_data.get_cam_config('down_stairs')
    cam_sys_config = cam_data.get_sys_config()

    feed_url = (f'{cam_config["protocol"]}://'
                f'{cam_sys_config["username"]}:'
                f'{cam_sys_config["password"]}@'
                f'{cam_config["url"]}:'
                f'{cam_config["port"]}/'
                f'{cam_sys_config["endpoint"]}')

    capture = cv2.VideoCapture(feed_url)
    has_frame, frame = capture.read()

    fh = frame.shape[0]
    fw = frame.shape[1]

    LOGGER.info(f'Original Frame Shape: {fw} x {fh}')

    # Variables to Hold Inference Time Information
    total_ag_inference_time = 0
    inferred_face_count = 0

    while has_frame:
        resized = cv2.resize(frame, (W, H))
        resized = resized.transpose((2, 0, 1))
        input_image = resized.reshape((N, C, H, W))

        # Start Inference
        fdetect_start = time.time()
        results = face_detection_executable.infer(
            inputs={face_detection_input_layer: input_image})
        fdetect_end = time.time()
        inf_time = fdetect_end - fdetect_start
        fps = 1. / inf_time

        # Write Information on Image
        text = 'Face Detection - FPS: {}, INF: {}'.format(
            round(fps, 2), round(inf_time, 4))
        cv2.putText(frame, text, (0, 20), cv2.FONT_HERSHEY_COMPLEX,
                    0.6, (0, 125, 255), 1)

        # Print Bounding Boxes on Image
        detections = results[face_detection_output_layer][0][0]

        face_count = 0
        face_coordinates = list()
        face_frames = list()

        # Check All Detections
        for detection in detections:
            if detection[2] > FACE_DETECTION_THRESHOLD:

                # Crop Frame
                xmin = int(detection[3] * fw)
                ymin = int(detection[4] * fh)
                xmax = int(detection[5] * fw)
                ymax = int(detection[6] * fh)

                coordinates = [xmin, ymin, xmax, ymax]
                face_coordinates.append(coordinates)

                face = crop_frame(
                    frame=frame, coordinate=coordinates, normalized=False)

                r_frame = cv2.resize(face, (WAG, HAG))
                r_frame = cv2.cvtColor(r_frame, cv2.COLOR_BGR2RGB)
                r_frame = np.transpose(r_frame, (2, 0, 1))
                r_frame = np.expand_dims(r_frame, axis=0)

                if NAG > 1:
                    age_detection_input[face_count - 1:face_count, ] = r_frame
                else:
                    face_frames.append(r_frame)

                face_count += 1
                cv2.rectangle(frame, (xmin, ymin),
                              (xmax, ymax), (0, 125, 255), 3)

        agdetect_start = time.time()
        inferred_face_count += face_count

        if face_count > 0 and NAG > 1:
            if AG_DYNAMIC_BATCH:
                age_gender_detection_executable.requests[0].set_batch(
                    face_count)

            age_gender_detection_executable.infer(
                {age_gender_detection_input_layer: age_detection_input})

            for f in range(face_count):
                age = int(
                    age_gender_detection_executable.requests[0].outputs[age_gender_detection_output_layers[0]][0][0][0][0] * 100)

                gender = 'male'
                if age_gender_detection_executable.requests[0].outputs[age_gender_detection_output_layers[1]][0][0][0][0] > \
                        age_gender_detection_executable.requests[0].outputs[age_gender_detection_output_layers[1]][0][1][0][0]:
                    gender = 'female'

                text = "A: {} - G: {}".format(age, gender)
                cv2.putText(frame, text, (face_coordinates[f][0], face_coordinates[f]
                                          [1] - 7), cv2.FONT_HERSHEY_PLAIN, 0.8, (0, 125, 255), 1)

            agdetect_end = time.time()

            # Write Information on Image
            inf_time = (agdetect_end - agdetect_start) / face_count
            fps = face_count / inf_time

        elif face_count > 0:
            f = 0
            for face in face_frames:
                age_gender_detection_executable.infer(
                    {age_gender_detection_input_layer: face})
                age = int(
                    age_gender_detection_executable.requests[0].outputs[age_gender_detection_output_layers[0]][0][0][0][0] * 100)

                gender = 'male'
                if age_gender_detection_executable.requests[0].outputs[age_gender_detection_output_layers[1]][0][0][0][0] > age_gender_detection_executable.requests[0].outputs[age_gender_detection_output_layers[1]][0][1][0][0]:
                    gender = 'female'

                text = "A: {} - G: {}".format(age, gender)
                cv2.putText(
                    frame, text, (face_coordinates[f][0], face_coordinates[f][1] - 7), cv2.FONT_HERSHEY_PLAIN, 0.8, (0, 125, 255), 1)

                f += 1

            agdetect_end = time.time()

            # Write Information on Image
            inf_time = (agdetect_end - agdetect_start) / f
            fps = f / inf_time

        if face_count > 0:
            text = 'AG Detection - FPS: {}, INF Per Face: {}'.format(
                round(fps, 2), round(inf_time, 4))
            cv2.putText(frame, text, (0, 40),
                        cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 125, 255), 1)
            total_ag_inference_time += inf_time

        cv2.imshow('Window', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        has_frame, frame = capture.read()

    LOGGER.info(f'Total AG Inference Time: {total_ag_inference_time}')
    LOGGER.info(f'Number of Face Inferred: {inferred_face_count}')
    LOGGER.info(
        f'Average AG Inference Time: {total_ag_inference_time / inferred_face_count}')
