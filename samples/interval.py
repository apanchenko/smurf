from enum import Enum, unique


# Binance interval
@unique
class Interval(Enum):
    _1m = '1m'
    _2m = '3m'
    _5m = '5m'
    _15m = '15m'
    _30m = '30m'
    _1h = '1h'
    _2h = '2h'
    _4h = '4h'
    _6h = '6h'
    _8h = '8h'
    _12h = '12h'
    _1d = '1d'
    _3d = '3d'
    _1w = '1w'

    # Minutes per interval
    @property
    def minutes(self):
        return {
            "m": 1,
            "h": 60,
            "d": 24 * 60,
            "w": 7 * 24 * 60,
        }.get(self.value[-1]) * int(self.value[:-1])

    # Seconds per interval
    @property
    def seconds(self):
        return self.minutes * 60