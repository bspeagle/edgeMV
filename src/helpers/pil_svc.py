"""
PIL helper modules.
"""

import io
import cv2
import numpy
from PIL import Image, ImageDraw
from helpers import utilities
from helpers.utilities import LOGGER


class ImageProcessing:
    """
    Functions to format and draw image data.
    """

    def __init__(self):
        self.current_frame = 0

    def format_img(self, frame):
        """
        Format the img for rekognition processing.
        """
        try:
            pil_frame = Image.fromarray(frame)
            stream = io.BytesIO()
            pil_frame.save(stream, format='JPEG')
            jpeg_frame = stream.getvalue()

            return jpeg_frame

        except Exception as ex:
            LOGGER.error(ex)
            raise ex

    def make_it_cool(self, face_details, jpeg_frame):
        """
        Draw bounding box and add meta to image.
        """

        stream = io.BytesIO(jpeg_frame)
        image = Image.open(stream)

        img_width, img_height = image.size

        img_draw = ImageDraw.Draw(image)

        for face in face_details:
            left = img_width * float(face['BoundingBox']['Left'])
            top = img_height * float(face['BoundingBox']['Top'])
            width = img_width * float(face['BoundingBox']['Width'])
            height = img_height * float(face['BoundingBox']['Height'])

            points = (
                (left, top),
                (left + width, top),
                (left + width, top + height),
                (left, top + height),
                (left, top)
            )

            img_draw.line(points, fill='#00d400', width=2)

        del img_draw

        merged_img = self.draw_frame_stamp(jpeg_frame, image)

        return merged_img

    def draw_frame_stamp(self, jpeg_frame, image=None):
        """
        Add the current frame count to the outgoing frame.
        """

        self.current_frame += 1

        frame_msg = f'Current_Frame: {self.current_frame}'

        jpeg_stream = io.BytesIO(jpeg_frame)
        jpeg_stream_image = Image.open(jpeg_stream)

        img_width, img_height = jpeg_stream_image.size

        if image is None:
            background_frame = jpeg_stream_image.convert('RGBA')
        else:
            image_stream = io.BytesIO()
            image.save(image_stream, format='JPEG')
            background_image_jpeg_frame = Image.open(image_stream)
            background_frame = background_image_jpeg_frame.convert('RGBA')

        fresh_frame = Image.new(
            'RGBA', background_frame.size, (255, 255, 255, 0))

        img_draw = ImageDraw.Draw(fresh_frame)
        font_width, font_height = img_draw.textsize(frame_msg)

        img_draw.text(((img_width-font_width)/2, (img_height-font_height)/2),
                      frame_msg, fill=(255, 255, 255, 128))

        merged_frame = Image.alpha_composite(background_frame, fresh_frame)
        opencv_merged_frame = cv2.cvtColor(
            numpy.array(merged_frame), cv2.COLOR_RGB2BGR)

        del img_draw

        return opencv_merged_frame
