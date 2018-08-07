import logging
import time
import os
import json
from datetime import datetime
from binance_api import Binance

broker = Binance(
    API_KEY='',
    API_SECRET=''
)

# setup ligging
logging.basicConfig(
    format="%(asctime)s [%(levelname)-5.5s] %(message)s",
    level=logging.DEBUG,
    handlers=[
        logging.FileHandler("{path}/logs/{fname}.log".format(path=os.path.dirname(os.path.abspath(__file__)), fname="binance")), logging.StreamHandler()
    ])
log = logging.getLogger('')

log.info("-------------------------------------------------------------------------------")
broker.ping()

exchangeInfo = broker.exchangeInfo()
log.info(exchangeInfo)

with open('data/binance.json', 'w') as outfile:
    json.dump(exchangeInfo, outfile)