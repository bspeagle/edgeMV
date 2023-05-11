"""
OpenCV module for processing video feed.
"""

from io import BytesIO
import os
import time
import imutils
from imutils.video import WebcamVideoStream
import numpy
import cv2
from helpers import dev_ops, pil_svc, utilities
from helpers.mongo import MongoQueues
from helpers.utilities import Extras, LOGGER

LOGGER.debug(cv2.getBuildInformation())


class Video:
    """
    Video stream processing and frame read, send to buffer.
    """

    def __init__(self):
        self.app = os.getenv('APP')
        self.local_ops = dev_ops.LocalOps()
        self.cam_data = Extras.CameraData()
        self.cam_data.load_cam_data(os.getenv('FOSCAM_DATA_FILE'))
        self.pil = pil_svc.ImageProcessing()
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml')
        self.cam_stream = WebcamVideoStream

        cv2.useOptimized()

        if os.getenv('APP_ENV').upper() == 'DEV':
            self.local_ops.create_dirs()

    def __start_cam_stream(self):
        """
        Start the cam stream and wait 2 seconds.
        Just in case...
        """

        cam_config = self.cam_data.get_cam_config('down_stairs')
        cam_sys_config = self.cam_data.get_sys_config()

        feed_url = (f'{cam_config["protocol"]}://'
                    f'{cam_sys_config["username"]}:'
                    f'{cam_sys_config["password"]}@'
                    f'{cam_config["url"]}:'
                    f'{cam_config["port"]}/'
                    f'{cam_sys_config["endpoint"]}')

        LOGGER.debug(f'Feed_url: {feed_url}')
        LOGGER.info('Starting cam stream...')

        self.cam_stream = WebcamVideoStream(src=feed_url).start()
        time.sleep(2.0)

        # # Get video stream size. Not needed yet. Just didn't want to forget about it.
        # width  = cam_stream.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)
        # height = cam_stream.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)

    def process_video(self):
        """
        Reading frames... And don't do drugs!
        """

        self.__start_cam_stream()

        inbound_queue = MongoQueues.Inbound()
        current_frame = 0
        frame_skip = int(os.getenv('FRAME_SKIP'))

        while True:
            frame = self.cam_stream.read()

            if current_frame % frame_skip == 0:
                frame = imutils.resize(
                    frame,
                    width=int(os.getenv('FRAME_WIDTH')),
                    height=int(os.getenv('FRAME_HEIGHT'))
                )

                LOGGER.debug(f'Current incoming frame: {current_frame}')

                inbound_queue.add_frame(frame)

            current_frame += 1

    def cv2_detect_faces(self, frame):
        """
        Nothing fancy to see here. A quick face check to see if
        the image needs to go through Rekognition. Marks the image
        and sends it to the in-memory mongo db.
        """

        LOGGER.debug(f'jpeg_frame: {frame}')

        cv2_detection = bool

        load_bytes = BytesIO(frame.payload['frame'])
        loaded_np = numpy.load(load_bytes, allow_pickle=True)

        frame_gray = cv2.cvtColor(loaded_np, cv2.COLOR_BGR2GRAY)
        frame_gray = cv2.equalizeHist(frame_gray)

        faces = self.face_cascade.detectMultiScale(
            frame_gray,
            scaleFactor=1.3,
            minNeighbors=3,
            minSize=(30, 30)
        )

        if len(faces) > 0:
            LOGGER.debug(f'Found {len(faces)} Faces.')
            cv2_detection = True
        else:
            cv2_detection = False

        return cv2_detection
