"""
Stream tools. I'm going crazy!!!!!!
"""

from io import BytesIO
import os
from time import sleep
import cv2
import numpy
from helpers import utilities
from helpers.mongo import MongoQueues
from helpers.utilities import LOGGER


def generate_media_response():
    """
    Generate media response for outgoing stream.
    """

    outbound_queue = MongoQueues.Outbound()
    stream_pause = float(os.getenv('STREAM_PAUSE'))

    while True:
        next_frame = outbound_queue.get_next_frame()
        if next_frame is None:
            continue
        else:
            load_bytes = BytesIO(next_frame.payload['frame'])
            loaded_np = numpy.load(load_bytes, allow_pickle=True)

            (flag, encoded_image) = cv2.imencode(
                ".jpg", loaded_np)

            if not flag:
                continue

            yield b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encoded_image) + b'\r\n'

            sleep(stream_pause)
