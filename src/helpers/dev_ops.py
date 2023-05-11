"""
File system automation for temp storage
"""

import os
import shutil
from helpers import utilities
from helpers.utilities import LOGGER


class LocalOps:
    """
    Local temp storage.
    """

    def __init__(self):
        self.local_path = os.getenv('LOCAL_TEMP_DIR').upper()
        self.frames_dir = os.getenv('FRAMES_DIR').upper()
        self.faces_dir = os.getenv('FACES_DIR').upper()

    def create_dirs(self):
        """
        Create local temp storage directories.
        """

        if os.path.exists(self.local_path):
            shutil.rmtree(self.local_path)
        else:
            os.makedirs(self.local_path)

        if not os.path.exists(f'{self.local_path}{self.frames_dir}'):
            os.makedirs(f'{self.local_path}{self.frames_dir}')

        if not os.path.exists(f'{self.local_path}{self.faces_dir}'):
            os.makedirs(f'{self.local_path}{self.faces_dir}')

        LOGGER.info('Local storage directores created!')

    def delete_dirs(self):
        """
        Delete local temp storage directories.
        """

        if os.path.exists(self.local_path):
            shutil.rmtree(self.local_path)
            LOGGER.info('Local storage directories deleted.')
        else:
            LOGGER.warning('Local storage directories do not exist.')
