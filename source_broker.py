import logging
import time
import os
import json
from datetime import datetime
from binance_api import BinanceApi

# bridge between API and Smurfy
class SourceBroker():

    def __init__(self, name):
        self.name = name
        self.api = BinanceApi(API_KEY='', API_SECRET='')

    def exchangeInfo(self):
      return self.api.exchangeInfo()