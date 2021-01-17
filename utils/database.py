"""
Handle MongoDB communication
"""

import json
from pymongo import MongoClient

# load config settings
config_file = open("config.json", "r")
config = json.load(config_file)
mongo_connection_string = config["MONGO_CONNECTION_STRING"]

# connect to MongoDB
client = MongoClient(mongo_connection_string)

# initialize (if necessary) and get watchlists collection
db = client["BrokerBot"]
lists_col = db["watchlists"]


def _get_unique(event):
    """ generate unique id string for each user from slack's team id and user id """

    user_id = event.get("user")
    team_id = event.get("team")
    return team_id + "-" + user_id


def new_user(event):
    """ create document for a new user """

    id = _get_unique(event)

    lists_col.insert_one({"_id": id, "watchlist": ["AAPL", "TSLA", "MSFT"]})


def get_watchlist(event):
    """ retrieve a user's watchlist from MongoDB """

    id = _get_unique(event)

    try:
        return lists_col.find_one({"_id": id})["watchlist"]
    except:
        # user is not registered
        new_user(event)
        get_watchlist(event)


def add(tickers, event):
    """ update a user's watchlist in MongoDB with added tickers """

    id = _get_unique(event)

    try:
        watchlist = lists_col.find_one({"_id": id})["watchlist"]
    except:
        # user is not registered
        new_user(event)
        add(tickers, event)

    # arrays to keep track of results
    added = []
    existing = []

    for ticker in tickers:
        if ticker not in watchlist:
            watchlist.append(ticker)
            added.append(ticker)
        else:
            existing.append(ticker)

    # if tickers were added, sort and update watchlist
    if added:
        watchlist.sort()
        lists_col.find_one_and_update({"_id": id}, {"$set": {"watchlist": watchlist}})

    return added, existing


def remove(tickers, event):
    """ update a user's watchlist in MongoDB with removed tickers """

    id = _get_unique(event)

    try:
        watchlist = lists_col.find_one({"_id": id})["watchlist"]
    except:
        # user is not registered
        new_user(event)
        remove(tickers, event)

    # arrays to keep track of results
    removed = []
    not_found = []

    for ticker in tickers:
        if ticker in watchlist:
            watchlist.remove(ticker)
            removed.append(ticker)
        else:
            not_found.append(ticker)

    # if tickers were removed, update watchlist
    if removed:
        lists_col.find_one_and_update({"_id": id}, {"$set": {"watchlist": watchlist}})

    return removed, not_found
