"""
Logging module.
"""

import logging
import os

APP = os.getenv('APP')
ENV = os.getenv('APP_ENV').upper()
LOG_FORMAT = '%(levelname)s: %(module)s: %(funcName)s: %(message)s'
LOG_LVL = ''

if ENV == 'DEV':
    LOG_LVL = logging.DEBUG
elif ENV == 'CASUAL':
    LOG_LVL = logging.INFO
elif ENV == 'PROD':
    LOG_LVL = logging.ERROR

logging.basicConfig(level=LOG_LVL, format=LOG_FORMAT)

LOGGER = logging.getLogger(APP)