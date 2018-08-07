import time
from datetime import datetime, timezone
from binance.client import Client
from store import Store

MAX_LIMIT = 500


def convert_datetime(dt):
    assert isinstance(dt, datetime)
    return int(dt.replace(tzinfo=timezone.utc).timestamp()) # to UTC


def get_candles(symbol, interval, start_datetime, end_datetime):
    start = convert_datetime(start_datetime)
    end = convert_datetime(end_datetime)

    data = []
    while start < end:
        # open store
        store = Store(symbol, interval, start);

        # fill store if empty
        if not store.exist():
            store_data = []
            client = Client("", "")
            count = store.candles_count
            while count > 0:
                limit = min(count, MAX_LIMIT)
                new_data = client.get_klines(symbol=symbol, interval=interval.value,
                                             limit=limit, startTime=start*1000)
                store_data += new_data
                start += len(new_data) * interval.seconds
                count -= len(new_data)
                time.sleep(1) # sleep to be kind to the API
            store.write(store_data)

        # read data from existing store
        data += store.read(end)
        start = store.next_start

    return data
