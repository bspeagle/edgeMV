"""
BLAH BLAH BLAH
"""

from helpers import utilities
from helpers.mongo import MongoClient
from helpers.utilities import LOGGER


def test_task():
    LOGGER.info('test')


def crazy_task():
    LOGGER.info('crazy!')


def nope():
    LOGGER.info('nope :(')


def yeah_task():
    LOGGER.info('yeeeaaaahhhhhh!')


# module_stuff = dir()
# module_funcs = []
possibles = globals().copy()

for thing in possibles:
    if thing.endswith('_task'):
        method = possibles.get(thing)
        method()
