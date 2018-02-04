import json
import candles
from binance.client import Client
from binance.helpers import date_to_milliseconds, interval_to_milliseconds
import csv

symbol = "ETHBTC"
start = "1 Nov, 2017"
end = "1 Dec, 2017"
interval = Client.KLINE_INTERVAL_30MINUTE

ticks = candles.get_historical_klines(symbol, interval, start, end)

# open a file with filename including symbol, interval and start and end converted to milliseconds
with open(
    "data/binance_{}_{}_{}.csv".format(
        symbol, interval, date_to_milliseconds(start)
    ), 'w'
) as csvfile:
    header = ['open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time']
    writer = csv.DictWriter(csvfile, fieldnames=header)
    writer.writeheader()

    for tick in ticks[:]:
        row = {
                    'open_time': tick[0],
                    'open':      tick[1],
                    'high':      tick[2],
                    'low':       tick[3],
                    'close':     tick[4],
                    'volume':    tick[5],
                    'close_time':tick[6]
                    # Quote asset volume
                    # Number of trades
                    # Taker buy base asset volume
                    # Taker buy quote asset volume
                    # Can be ignored
        }
        writer.writerow(row)