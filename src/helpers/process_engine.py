"""
Functions for different threads used by app.
"""

import asyncio
import atexit
from time import sleep
import ray
from helpers import flask_svc, utilities
from helpers.mongo import MongoQueues
from helpers.open_cv import Video
from helpers.rekognition_svc import BotoClient
from helpers.utilities import LOGGER


@ray.remote
def process_1_task():
    """
    Step 1
    """

    opencv_video = Video()
    opencv_video.process_video()


@ray.remote
def process_2_task():
    """
    Step 2
    """

    opencv_video = Video()
    inbound_queue = MongoQueues.Inbound()
    mezzo_queue = MongoQueues.Mezzo()
    mezzo_queue.clear()

    while True:
        next_frame = inbound_queue.get_next_frame()

        if next_frame is None:
            continue
        else:
            cv2_detection = opencv_video.cv2_detect_faces(next_frame)

            mezzo_queue.add_frame(next_frame, cv2_detection)


@ray.remote
def process_3_task():
    """
    Step 3
    """

    mezzo_queue = MongoQueues.Mezzo()
    outbound_queue = MongoQueues.Outbound()
    outbound_queue.clear()

    rekognition = BotoClient()

    while True:
        next_frame = mezzo_queue.get_next_frame()

        if next_frame is None:
            continue
        else:
            LOGGER.debug(f'next_frame: {next_frame}')

            rekognition.detect_and_format(next_frame)


def stat_bot():
    """
    Report queue counts WHEN I TELL IT TO!!!!
    """

    inbound_queue = MongoQueues.Inbound()
    mezzo_queue = MongoQueues.Mezzo()
    outbound_queue = MongoQueues.Outbound()

    sleep(15)

    while True:
        try:
            LOGGER.info(f"""
                        Stat bot:
                            -- Queue Counts --
                            inbound queue:  {inbound_queue.size()}
                            mezzo queue:    {mezzo_queue.size()}
                            outbound queue: {outbound_queue.size()}
                        """)

            sleep(10)
        except Exception as ex:
            LOGGER.error(ex)
            raise ex


def flask_app():
    """
    Start Flask server bitches!
    """

    try:
        flask_svc.flask_app()
    except Exception as ex:
        LOGGER.error(f'Flask task: {ex}')
        raise ex


def asyncio_end():
    """
    Find and cancel asyncio running tasks.
    """

    LOGGER.info('DONE!')


def background(function):
    """
    Asynchronous wrapper for ray calls (hopefully).
    """

    def wrapped(*args, **kwargs):
        return asyncio.get_event_loop().run_in_executor(None, function, *args, **kwargs)

    return wrapped


@background
def do_thing_somewhere_else(app_method):
    """
    Function to run ray.get() in the background (maybe).
    """

    ray.get(app_method.remote())


POSSIBLES = globals().copy()
atexit.register(asyncio_end)
# ASYNCIO_LOOP = asyncio.get_running_loop()
