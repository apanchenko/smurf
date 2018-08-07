import candles
from binance.client import Client
from binance.helpers import *
from interval import Interval

start = "1 Nov, 2017"
end = "1 Dec, 2017"
interval = Client.KLINE_INTERVAL_30MINUTE

data = candles.get_candles("ETHBTC", Interval._15m, datetime(2018, 1, 1), datetime(2018, 1, 3))

print('Candles count', len(data))