"""
Global settings for edgeMV.
"""

import json
import re
import statistics
from helpers import startup
from helpers.logger_svc import LOGGER

class Extras:
    """
    Extra utility functions for random shit.
    """

    class CameraData:
        """
        Load config values from camera data files.
        """

        def __init__(self):
            self.cam_data = ''

        def load_cam_data(self, file,):
            """
            Read cam data file to load cam configs and return json object.
            """

            with open(file, newline='') as data_file:
                self.cam_data = json.loads(data_file.read())

        def get_all_data(self):
            """
            Return the loaded cam data file.
            """

            return self.cam_data[0]

        def get_cam_config(self, cam_name):
            """
            Get cam config object from cam name.
            """

            for cam in self.cam_data[0]['cam_data']:
                if cam['name'] == cam_name:
                    return cam

        def get_sys_config(self):
            """
            Get system config object.
            """

            config_data = self.cam_data[0]['system_data'][0]

            return config_data

    def get_median_age(self, age_range):
        """
        Get the median age of provided min/max ages
        from Rekogniton response.
        """

        start = age_range['Low']
        end = age_range['High']
        num_range = []

        while start < end:
            num_range.append(start)
            start += 1

        return statistics.median(num_range)

    def replace_chars(self, eval_str, find_str, rplc_str):
        """
        Locate characters in string and replace with
        new string. DESTROY. ALL. HUMAN(S).
        """

        new_str = re.sub(fr'({find_str})', fr'{rplc_str}\1', eval_str)

        return new_str
