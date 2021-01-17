"""
Handle slack API calls
"""

import logging
import json

# logging.basicConfig(level=logging.DEBUG)

from slack import WebClient
from slackeventsapi import SlackEventAdapter
from slack.errors import SlackApiError

# load config settings
config_file = open("config.json", "r")
config = json.load(config_file)

slack_oauth_token = config["SLACK_OAUTH_TOKEN"]
slack_bot_oauth_token = config["SLACK_BOT_OAUTH_TOKEN"]
slack_signing_secret = config["SLACK_SIGNING_SECRET"]

# Slack setup
userClient = WebClient(token=slack_oauth_token)
botClient = WebClient(token=slack_bot_oauth_token)
BOT_ID = botClient.api_call("auth.test")["user_id"]
BOT_NAME = botClient.api_call("auth.test")["user"]


def sendReply(event, message, attachments=None):
    """ send message as a reply to user's message """

    channel_id = event.get("channel")
    ts = event.get("ts")

    try:
        response = botClient.chat_postMessage(
            channel=channel_id, thread_ts=ts, text=message, attachments=attachments
        )
        return response

    except SlackApiError:
        assert response["error"]


def uploadFile(event, filename):
    """ upload a file to Slack """

    channel_id = event.get("channel")
    ts = event.get("ts")

    try:
        response = userClient.files_upload(
            channel=channel_id,
            thread_ts=ts,
            file=filename,
            filetype="png",
            initial_comment="filedjhfdsf",
        )

        return response

    except SlackApiError:
        assert response["error"]


def publicURL(id):
    """ make a file public and return the URL """

    try:
        response = userClient.files_sharedPublicURL(file=id)

        return response["file"]["permalink_public"]

    except SlackApiError:
        assert response["error"]

        # curl -F file=@temp/risk_vs_return.png -F "initial_comment=I play the drums." -F channels=C024BE91L -F thread_ts=1532293503.000001 -H "Authorization: Bearer xoxp-xxxxxxxxx-xxxx" https://slack.com/api/files.upload