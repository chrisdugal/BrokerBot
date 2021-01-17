"""
Process bot commands
"""

from utils import messages, database, analysis
import os


def process(event):
    """ process incoming app mentions to check for valid commands """

    if event.get("text").upper().__contains__("LIST"):
        handle_list(event)
    elif event.get("text").upper().__contains__("ADD"):
        handle_add(event)
    elif event.get("text").upper().__contains__("REMOVE"):
        handle_remove(event)
    elif event.get("text").upper().__contains__("PREDICT"):
        handle_predict(event)
    elif event.get("text").upper().__contains__("RISK-RETURN"):
        handle_risk_return(event)
    elif event.get("text").upper().__contains__("SMA"):
        handle_sma(event)
    elif event.get("text").upper().__contains__("HELP"):
        handle_help(event)
    else:
        messages.error(event)


def clean_args(args):
    """ remove potential links for tickers with a "." """
    return [
        arg[arg.index("|") + 1 : -1] if arg.__contains__("<HTTP://") else arg
        for arg in args
    ]


def process_args(cmd, event):
    """ process and return tickers from command arguments """

    # get arguments from message text
    text = event.get("text")
    args = text.upper().replace(",", "").split()
    args = args[args.index(cmd) + 1 :]

    # check for no args
    if not args:
        messages.invalid_args(event)
        return

    # check for SMA invalid args
    periods = args[-1]
    if cmd == "SMA":
        args.remove(periods)
        if not args or not periods.__contains__("/"):
            messages.invalid_args(event)
            return
        else:
            try:
                short_sma = int(periods.split("/")[0])
                long_sma = int(periods.split("/")[1])
                if short_sma >= long_sma:
                    raise Exception
            except:
                messages.invalid_args(event)
                return

    # clean args
    args = clean_args(args)

    # lists to keep track of results
    tickers = []
    invalid = []

    # filter out invalid tickers
    for arg in args:
        if analysis.is_valid_ticker(arg):
            tickers.append(arg)
        else:
            invalid.append(arg)

    # return results
    if cmd == "SMA":
        return tickers, invalid, short_sma, long_sma

    return tickers, invalid


def handle_list(event):
    """ list ticker(s) in user's watchlist """

    # get watchlist and compose string list
    watchlist = database.get_watchlist(event)
    tickers = "\n".join(watchlist) if watchlist else None

    # send message
    messages.show_watchlist(tickers, event)


def handle_add(event):
    """ add ticker(s) to user's watchlist """

    # get tickers
    tickers, invalid = process_args("ADD", event)

    # add to database
    added, existing = database.add(tickers, event)

    # send message
    messages.show_added(added, existing, invalid, event)


def handle_remove(event):
    """ remove ticker(s) from user's watchlist """

    # get tickers
    tickers, invalid = process_args("REMOVE", event)

    # remove from database
    removed, not_found = database.remove(tickers, event)

    # send message
    messages.show_removed(removed, not_found, invalid, event)


def handle_predict(event):
    """ get ML model predictions for ticker """

    # get tickers
    tickers, invalid = process_args("PREDICT", event)

    svm_preds = []
    lr_preds = []
    ann_preds = []

    # generate predictions
    for ticker in tickers:
        svm_pred, lr_pred = analysis.svm_prediction(ticker)
        ann_pred = analysis.ann_prediction(ticker)

        svm_preds.append(svm_pred)
        lr_preds.append(lr_pred)
        ann_preds.append(ann_pred)

    # compose results string
    results = []
    for (ticker, svm_pred, lr_pred, ann_pred) in zip(
        tickers, svm_preds, lr_preds, ann_preds
    ):
        bid_ask = analysis.get_ticker_bid_ask(ticker)
        results.append(
            f"{ticker} buy-ask: {bid_ask}\n"
            f"\tsvm: {svm_pred[0]}, confidence: {svm_pred[1]}%\n"
            f"\tlr: {lr_pred[0]}, confidence: {lr_pred[1]}%\n"
            f"\tann: {ann_pred[0]}, confidence: {ann_pred[1]}%\n"
        )
    result = "\n" + "\n".join(results)

    # send message
    messages.show_predictions(result, invalid, event)


def handle_risk_return(event):
    """ get risk vs return plot for ticker(s) """

    # get tickers
    tickers, invalid = process_args("RISK-RETURN", event)

    # generate risk vs return plot
    filename = analysis.risk_vs_return(tickers)

    # send message
    messages.show_risk_return(filename, tickers, invalid, event)

    # remove file
    os.remove(filename)


def handle_sma(event):
    """ check for custom sma crossover for ticker(s) """

    # get tickers
    tickers, invalid, short_sma, long_sma = process_args("SMA", event)

    # get sma crossover status for each ticker
    sma_status = [
        analysis.sma_crossover(ticker, short_sma, long_sma) for ticker in tickers
    ]

    # compose result string
    results = []
    for (ticker, sma) in zip(tickers, sma_status):
        results.append(f"{ticker}: {sma}")
    result = "\n" + "\n".join(results)

    # send message
    messages.show_sma_results(result, invalid, event)


def handle_help(event):
    """ give user instructions and a command list """
    messages.help(event)
