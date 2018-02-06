from binance.client import Client
from binance.helpers import *
import time
from store import Store

MAX_LIMIT = 500


def get_candles(symbol, interval, start, end):
    assert isinstance(start, datetime)
    assert isinstance(end, datetime)

    # create store
    store = Store(symbol, interval, start);

    # fill store if empty
    if not store.exist():
        store_data = []
        client = Client("", "")
        count = store.candles_count()
        start_ts = start.timestamp() * 1000
        while count > 0:
            limit = min(count, MAX_LIMIT)
            data = client.get_klines(symbol=symbol, interval=interval.value,
                                     limit=limit, startTime=int(start_ts))
            assert len(data) == limit
            store_data += data
            count -= limit
            start_ts += limit * interval.ms_per_candle()

            # sleep to be kind to the API
            time.sleep(1)

        store.write(data)

    # read data
    data = store.read(end)
    return data
