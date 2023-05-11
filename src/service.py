"""
Main service to do all the things.
"""

import ray
from helpers import mongo, process_engine, utilities
from helpers.utilities import LOGGER


def main():
    """
    Do I really need to say it? It's the main function. WTF?
    """

    try:
        ray.init()
        mongo.db_init()

        for thing in process_engine.POSSIBLES:
            if thing.endswith('_task'):
                method = process_engine.POSSIBLES.get(thing)
                process_engine.do_thing_somewhere_else(method)

        process_engine.flask_app()

    except Exception as ex:
        LOGGER.exception(ex)
        exit()
        raise ex


main()
