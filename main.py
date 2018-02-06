import candles
from binance.client import Client
from binance.helpers import *
import csv
from interval import Interval

start = "1 Nov, 2017"
end = "1 Dec, 2017"
interval = Client.KLINE_INTERVAL_30MINUTE

ticks = candles.get_candles("ETHBTC", Interval._15m, datetime(2018, 1, 1), datetime(2018, 1, 2))

print('Candles ', len(ticks))