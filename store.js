var moment = require('moment'),
    binance = require('node-binance-api'),
    fs = require('fs');
require('dotenv').config();


binance.options({
  APIKEY: process.env.BINANCE_APIKEY,
  APISECRET: process.env.BINANCE_APISECRET,
  useServerTime: true, // If you get timestamp errors, synchronize to server time at startup
  test: true // If you want to use sandbox mode where orders are simulated
});



// Intervals: 1m,3m,5m,15m,30m,1h,2h,4h,6h,8h,12h,1d,3d,1w,1M
var Interval = Object.freeze({D:"1d", M:"1M"});
var Ticket = Object.freeze({BTCUSDT:"BNBBTC"});

var Data = class Data {

  constructor(ticket) {
    this.ticket = ticket;
  }

  get(interval) {
    var fileName = './data/' + this.ticket + '_' + interval + '.csv';
    if (fs.existsSync(fileName)) {
      
    }
    else {
      binance.candlesticks(this.ticket, interval, (error, ticks, symbol) => {
        console.log(this.ticket + ' receive ' + ticks.length + ' ticks');
//for (let tick of ticks)
//let [time, open, high, low, close, volume, closeTime, assetVolume, trades, buyBaseVolume, buyAssetVolume, ignored] = tick;
        var data = ticks.join("\n");
        fs.writeFile(fileName, data, err => {
          if (err) throw err;
          console.log(fileName + ' saved');
        });

      }, {limit: 10, endTime: 1514764800000});
    }
  } // get

} // class Data

exports.Interval = Interval;
exports.Ticket = Ticket;
exports.Data = Data;