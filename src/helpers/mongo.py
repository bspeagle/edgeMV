"""
Used to connect to MongoDB.
"""

from io import BytesIO
import json
import os
from uuid import uuid4
from mongo_queue.queue import Queue
import numpy
import pymongo
from helpers import utilities
from helpers.utilities import LOGGER


class MongoClient:
    """
    Mongo client init.
    """

    def __init__(self):
        self.__mongo_env = os.getenv('MONGO_ENV')
        self.__mongo_db = os.getenv('APP')
        self.client = pymongo.MongoClient(
            os.getenv('MONGO_LOCAL_HOST'), int(os.getenv('MONGO_LOCAL_PORT')))
        self.mongo_db = self.client.get_database(self.__mongo_db)
        self.mongo_collection = pymongo.collection.Collection
        self.queues = eval(os.getenv('MONGO_QUEUES'))

    def set_collection(self, index):
        """
        Set the collection to use.
        """

        self.mongo_collection = self.mongo_db.get_collection(
            self.queues[index])


class MongoQueues:
    """
    Mongo queues bitches!
    """

    class Inbound(Queue):
        """
        Inbound frames from video stream.
        """

        def __init__(self):
            self.mongo_client = MongoClient()
            self.mongo_client.set_collection(0)
            super().__init__(collection=self.mongo_client.mongo_collection,
                             consumer_id=os.getenv('APP'))

        def add_frame(self, frame):
            """
            Add a frame to the inbound queue from the video stream.
            """

            frame_bytes = BytesIO()
            numpy.save(frame_bytes, frame, allow_pickle=True)
            frame_bytes = frame_bytes.getvalue()

            frame_container = {
                "frame": frame_bytes
            }

            self.put(frame_container)

        def get_next_frame(self):
            """
            Get the next frame from the inbound queue.
            """
            try:
                next_frame = self.next()
                next_frame.complete()

                return next_frame

            except AttributeError as ex:
                if ex == "'NoneType' object has no attribute 'complete'":
                    LOGGER.warning('No frames returned. Moving on.')
                    return None

    class Mezzo(Queue):
        """
        Middle queue for frames passed through cv2 and awaiting rekognition.
        """

        def __init__(self):
            self.mongo_client = MongoClient()
            self.mongo_client.set_collection(1)
            super().__init__(collection=self.mongo_client.mongo_collection,
                             consumer_id=os.getenv('APP'))

        def add_frame(self, frame, cv2_detection):
            """
            Add a frame container to the mezza queue, from the inbound
            queue, with cv2_detection eval result.
            """

            frame_container = {
                'frame': frame.payload['frame'],
                'cv2_detection': cv2_detection
            }

            self.put(frame_container)

        def get_next_frame(self):
            """
            Get the next frame from the mezzo queue.
            """

            try:
                next_frame = self.next()
                next_frame.complete()

                return next_frame

            except AttributeError as ex:
                if ex == "'NoneType' object has no attribute 'complete'":
                    LOGGER.warning('No frames returned. Moving on.')
                    return None

    class Outbound(Queue):
        """
        Outbound queue. The final frontier...   That's StarsWar right?
        """

        def __init__(self):
            self.mongo_client = MongoClient()
            self.mongo_client.set_collection(2)
            super().__init__(collection=self.mongo_client.mongo_collection,
                             consumer_id=os.getenv('APP'))

        def add_frame(self, frame):
            """
            Add the final fucking frame. Fuck yeah!
            """

            frame_bytes = BytesIO()
            numpy.save(frame_bytes, frame, allow_pickle=True)
            frame_bytes = frame_bytes.getvalue()

            uuid = uuid4()

            frame_container = {
                "frame": frame_bytes,
                "uuid": uuid
            }

            self.put(frame_container)

            return uuid

        def get_next_frame(self):
            """
            Get the next frame from the mezzo queue.
            """

            try:
                next_frame = self.next()
                next_frame.complete()

                return next_frame

            except AttributeError as ex:
                if ex == "'NoneType' object has no attribute 'complete'":
                    LOGGER.warning('No frames returned. Moving on.')
                    return None

        def update_frame(self, pil_frame, frame_container):
            """
            Update a frame by uuid in the outbound queue.
            """

            try:
                frame_bytes = BytesIO()
                numpy.save(frame_bytes, pil_frame, allow_pickle=True)
                frame_bytes = frame_bytes.getvalue()

                response = self.collection.update_one({'payload.uuid': frame_container['uuid']}, {
                    '$set': {'payload.frame': frame_bytes}})

                LOGGER.debug(f"""
                            -- Update Response --
                            # of matched documents:                         {response.matched_count}
                            # of modified documents:                        {response.modified_count}
                            _id for upserted document (if upsert occurred): {response.upserted_id}
                            Acknowledged:                                   {response.acknowledged}
                            """)

            except Exception as ex:
                LOGGER.exception(ex)
                raise ex


def db_init():
    """
    Initialize the database. That sounded smart! Actually delete the db if it
    exists, create a new one and add collections. Cattle not pets!
    """

    mongo = MongoClient()
    database = os.getenv('APP')

    mongo.client.drop_database(database)

    mongo_db = mongo.client.get_database(os.getenv('APP'))

    for queue in mongo.queues:
        mongo_collection = mongo_db.get_collection(queue)
        response = mongo_collection.insert_one({'test': '123'})
        mongo_collection.delete_one({'_id': response.inserted_id})

    mongo_collection = mongo_db.get_collection(mongo.queues[2])
    mongo_collection.create_index('uuid')

    LOGGER.info(f"""
                -- MongoDB Info --
                Databases:          {mongo.client.list_database_names()}
                Collections/Queues: {mongo_db.list_collection_names()}
                """)
