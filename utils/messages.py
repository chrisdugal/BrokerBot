"""
Messages sent by BrokerBot
"""

from utils import slackapi as s


def help(event):
    """ send help message """

    s.sendReply(
        event,
        f"Use tickers as they appear on www.finance.yahoo.com.\n"
        f"Commands:\n"
        f"Show watchlist: @{s.BOT_NAME} list\n"
        f"Add to watchlist: @{s.BOT_NAME} add <ticker(s)>\n"
        f"Remove from watchlist: @{s.BOT_NAME} remove <ticker(s)>\n"
        f"Get predictions: @{s.BOT_NAME} predict <ticker(s)>\n"
        f"\tNote: might take some time if many tickers are provided\n"
        f"Get risk vs return plot: @{s.BOT_NAME} risk-return <ticker(s)>\n"
        f"Check for SMA crossover: @{s.BOT_NAME} sma <ticker(s)> <short>/<long>\n"
        f"Help: @{s.BOT_NAME} help",
    )


def error(event):
    """ send error message """

    s.sendReply(
        event,
        f'Sorry, I don\'t know what this means. Reminder: use "@{s.BOT_NAME} help" for help.',
    )


def invalid_args(event):
    """ send invalid arguments message """

    s.sendReply(
        event,
        f'Please provide the proper arguments. Use "@{s.BOT_NAME} help" for help.',
    )


def show_watchlist(tickers, event):
    """ send watchlist """

    s.sendReply(event, tickers if tickers else "Your watchlist is empty.")


def show_added(added, existing, invalid, event):
    """ send added tickers and potential issues """

    s.sendReply(
        event,
        f'Added: {", ".join(added) if added else "none"}\n'
        f'Existing: {", ".join(existing) if existing else "none"}\n'
        f'Invalid: {", ".join(invalid) if invalid else "none"}',
    )


def show_removed(removed, not_found, invalid, event):
    """ send removed tickers and potential issues """

    s.sendReply(
        event,
        f'Removed: {", ".join(removed) if removed else "none"}\n'
        f'Not found: {", ".join(not_found) if not_found else "none"}\n'
        f'Invalid: {", ".join(invalid) if invalid else "none"}',
    )


def show_predictions(result, invalid, event):
    """ send ML predictions for stock prices """

    s.sendReply(
        event,
        f'Predictions: {result if result else "none"}\n'
        f'Invalid: {", ".join(invalid) if invalid else "none"}',
    )

    if result:
        s.sendReply(
            event,
            "Note: The models do not take news or current events into account. "
            "Some models work better for different stocks based on their behaviour, "
            "try following the predictions for a while to determine which is best.",
        )


def show_risk_return(filename, tickers, invalid, event):
    """ send risk vs return plot and potential issues """

    if tickers:
        r = s.uploadFile(event, filename)
        url = s.publicURL(r["file"]["id"])

    s.sendReply(
        event,
        f'Invalid: {", ".join(invalid) if invalid else "none"}',
        attachments=[
            {
                "fallback": "risk vs return plot",
                "text": "risk vs return plot: " + ", ".join(tickers),
                "image_url": url,
            }
        ]
        if tickers
        else None,
    )


def show_sma_results(result, invalid, event):
    """ send sma results and potential issues """

    s.sendReply(
        event,
        f'Results: {result if result else "none"}\n'
        f'Invalid: {", ".join(invalid) if invalid else "none"}',
    )