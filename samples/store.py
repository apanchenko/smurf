import os
from datetime import date, datetime
from interval import Interval


class Store:
    HEADER = ['Open time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close time',
              'Quote asset volume', 'Number of trades',
              'Taker buy base asset volume', 'Taker buy quote asset volume', 'Can be ignored']

    def __init__(self, symbol, interval, start):
        assert isinstance(interval, Interval)
        assert isinstance(start, int)
        self.symbol = symbol
        self.interval = interval
        self.start = start

    # One day or one year
    def is_daily(self):
        return self.interval.value[-1] == "m"

    # Number of candles in complete file
    @property
    def candles_count(self):
        if self.is_daily():
            return int(60 * 24 / self.interval.minutes)

        year = datetime.utcfromtimestamp(self.start).date.year
        hours_per_year = (date(year=year) - date(year=year+1)) * 24
        hours_per_candle = self.interval.minutes / 60
        return int(hours_per_year / hours_per_candle)

    # Path to file
    @property
    def file_path(self):
        fmt = "data/binance_{}_{}_{}.csv"
        dt = datetime.utcfromtimestamp(self.start)
        if self.is_daily():
            return fmt.format(self.symbol, dt.strftime('%Y-%m-%d'), self.interval.value)
        return fmt.format(self.symbol, dt.date.year, self.interval.value)

    # Start time for following store
    @property
    def next_start(self):
        return self.start + (self.candles_count * self.interval.seconds)

    # Validate file
    def exist(self):
        return os.path.exists(self.file_path)

    # Read data from file
    def read(self, end):
        assert isinstance(end, int)
        assert end > self.start
        data = []
        with open(self.file_path) as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                if end > int(row[0]):
                    break
                data.append(row)
        return data

    # Write data
    def write(self, data):
        assert not self.exist()
        print('Write store', self.file_path)
        with open(self.file_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(data)
