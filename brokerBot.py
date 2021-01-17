"""
Startup Flask server and receive Slack events
"""

import logging

# logging.basicConfig(level=logging.DEBUG)


from utils import commands, slackapi
import threading
from slackeventsapi import SlackEventAdapter
from flask import Flask, Response

# Flask setup
app = Flask(__name__)
slack_event_adapter = SlackEventAdapter(
    slackapi.slack_signing_secret, "/slack/events", app
)


@slack_event_adapter.on("app_mention")
def message(payLoad):
    event = payLoad.get("event", {})

    # start thread to handle processing
    x = threading.Thread(target=commands.process, args=(event,))
    x.start()

    # send HTTP response
    return Response(), 200


# run on local port 5000
if __name__ == "__main__":
    app.run(debug=True, port=5000)