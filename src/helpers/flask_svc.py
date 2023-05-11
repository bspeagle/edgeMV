"""
Flask frontend and routing.
"""

import os
from flask import Flask
from flask import Response
from flask import render_template
from helpers import stream, utilities
from helpers.utilities import LOGGER

APP = Flask(
    __name__,
    template_folder='../templates',
    static_folder='../static'
)


def flask_app():
    """
    main function to run flask server.
    """

    APP.run(
        host=os.getenv('FLASK_IP'),
        port=os.getenv('FLASK_PORT'),
        debug=False,
        use_reloader=False
    )


@APP.route("/")
def index():
    """
    Return the rendered template.
    """

    return render_template("index.html")


@APP.route("/video_feed")
def video_feed():
    """
    Return the response generated along with
    the specific media type (mime type).
    """

    return Response(stream.generate_media_response(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")
