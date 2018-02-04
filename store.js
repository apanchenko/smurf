var moment = require('moment');
var stock = require('node-binance-api');
var fs = require('fs');
var csv = require("fast-csv");

const HEADERS = ['time', 'open', 'high', 'low', 'close', 'volume', 'closeTime', 'assetVolume', 'trades', 'buyBaseVolume', 'buyAssetVolume', 'ignored'];
const INTERVALS = ['1m','3m','5m','15m','30m','1h','2h','4h','6h','8h','12h','1d','3d','1w','1M'];

// data source options
stock.options({
  APIKEY: process.env.BINANCE_APIKEY,
  APISECRET: process.env.BINANCE_APISECRET,
  useServerTime: true, // If you get timestamp errors, synchronize to server time at startup
  test: true // If you want to use sandbox mode where orders are simulated
});



// Get candles.
//
function candles(symbol, interval, startMoment, endMoment, callback) {
  var path = fileName(symbol, startMoment, interval)
  if (fs.existsSync(path)) {
    csv
      .fromPath(path)
      .on("data", tick => callback(tick) );
      //.on("end", () => { console.log("done"); });
  }
  else {
    stock.candlesticks(symbol, interval, function onTicks(error, ticks, symbol) {
      if (error)
        throw error;
      console.log(symbol + ' receive ' + ticks.length + ' ticks');
      csv.writeToPath(path, ticks, {headers:HEADERS});
      for (var tick in ticks)
        callback(tick);
    }, {
      //limit: 10,
      startTime: startMoment.valueOf(),
      endTime: endMoment.valueOf()
    });
  }
}

// Get file name with data
//
function fileName(symbol, date, interval) {
  return './data/' + symbol + '/' + date.utc().format('YYYY-MM') + '_' + interval + '.csv';
}

// module exports
//
exports.INTERVALS = INTERVALS;
exports.candles = candles;