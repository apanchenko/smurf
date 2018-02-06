from interval import Interval
from datetime import date, datetime
import os
import csv


class Store:
    def __init__(self, symbol, interval, start):
        assert isinstance(interval, Interval)
        assert isinstance(start, datetime)
        self.symbol = symbol
        self.interval = interval
        self.start = start

    # One day or one year
    def is_daily(self):
        return self.interval.value[-1] == "m";

    # Number of candles in complete file
    def candles_count(self):
        if self.is_daily():
            return int(60 * 24 / self.interval.minutes_per_candle())

        year = self.start.date.year;
        hours_per_year = (date(year=year) - date(year=year+1)) * 24
        hours_per_candle = self.interval.minutes_per_candle() / 60
        return int(hours_per_year / hours_per_candle)

    # Path to file
    def file_path(self):
        fmt = "data/binance_{}_{}_{}.csv"
        if self.is_daily():
            return fmt.format(self.symbol, self.interval, self.start.strftime('%Y-%m-%d'))
        return fmt.format(self.symbol, self.interval, self.start.date.year)

    # Validate file
    def exist(self):
        return os.path.exists(self.file_path())

    # Read data from file
    def read(self, end):
        data = []
        # open a file with filename including symbol, interval and start and end converted to milliseconds
        with open(self.file_path()) as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                data += row
        return data

    # Write data
    def write(self, data):
        assert not self.exist()
        with open(self.file_path(), 'w') as csvfile:
            #fieldnames = ['first_name', 'last_name']
            #writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer = csv.writer(csvfile)
            #writer.writeheader()
            writer.writerows(data)