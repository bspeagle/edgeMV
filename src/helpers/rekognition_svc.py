"""
AWS Rekognition stuff. FIND FACES!
"""

from io import BytesIO
import threading
import boto3
import numpy
from helpers import pil_svc, utilities
from helpers.mongo import MongoQueues
from helpers.utilities import Extras, LOGGER


class BotoClient:
    """
    Functions to interact with AWS Rekognition.
    """

    def __init__(self):
        self.__rekognition = boto3.client('rekognition')
        self.__face_attrs = [
            'BoundingBox',
            'AgeRange',
            'EyeGlasses',
            'SunGlasses',
            'Gender',
            'Beard',
            'Emotions',
            'Confidence'
        ]
        self.__pil = pil_svc.ImageProcessing()
        self.extras = Extras()

    def __detect_faces(self, frame_container):
        """
        Takes frame_img and calls Rekognition API to get data meta.
        """

        response_data = []
        outbound_queue = MongoQueues.Outbound()

        load_bytes = BytesIO(frame_container.payload['frame'])
        loaded_np = numpy.load(load_bytes, allow_pickle=True)

        jpeg_frame = self.__pil.format_img(loaded_np)

        LOGGER.debug(f'frame_img: TYPE = {type(jpeg_frame)}')

        response = self.__rekognition.detect_faces(
            Image={
                'Bytes': jpeg_frame
            },
            Attributes=['ALL']
        )

        LOGGER.debug(f'Rekognition response: {response}')

        for face_data in response['FaceDetails']:
            this_face = {}

            LOGGER.debug(f'Face Data: {face_data}')

            for (key, val) in face_data.items():
                if key in self.__face_attrs:
                    LOGGER.debug(f'Item Details: [value: {type(val)} - {val}]')

                    if isinstance(val, dict):
                        if 'Low' and 'High' in val:
                            this_face.update(
                                {'Age': self.extras.get_median_age(val)})
                        elif 'Width' and 'Height' and 'Left' and 'Top' in val:
                            this_face.update({key: val})
                        else:
                            for (sub_key, sub_val) in val.items():
                                if sub_key == 'Value':
                                    this_face.update({key: sub_val})
                    elif isinstance(val, list) and key == 'Emotions':
                        this_face.update({key: val})
                    elif isinstance(val, float):
                        this_face.update({key: round(val, 2)})

            response_data.append(this_face)

        LOGGER.debug(f'Face Response Data: {response_data}')

        if response_data:
            LOGGER.debug(f'response_data: {response_data}')

            pil_frame = self.__pil.make_it_cool(
                response_data, jpeg_frame)

            outbound_queue.add_frame(pil_frame)

    def detect_and_format(self, frame_container):
        """
        Pulls frames with faces for Rekognition processing. Formats, detects
        and draws face stuff and then adds it back to the outgoing queue
        in same position.
        """

        outbound_queue = MongoQueues.Outbound()

        # response_uuid = outbound_queue.add_frame(pil_frame)

        if frame_container.payload['cv2_detection']:
            LOGGER.debug('cv2 detection found!')

            # new_frame_container = {
            #     "frame": frame_container.payload['frame'],
            #     "uuid": response_uuid
            # }

            aws_thread = threading.Thread(
                target=self.__detect_faces, args=(frame_container,), daemon=True)
            aws_thread.start()
            aws_thread.join()

        else:
            load_bytes = BytesIO(frame_container.payload['frame'])
            loaded_np = numpy.load(load_bytes, allow_pickle=True)

            jpeg_frame = self.__pil.format_img(loaded_np)

            pil_frame = self.__pil.draw_frame_stamp(jpeg_frame)

            outbound_queue.add_frame(pil_frame)
