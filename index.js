require('dotenv').config();
var moment = require('moment');
var store = require('./store.js');

const TICKET = 'BNBBTC';

store.candles(
  TICKET, '1m',
  moment('2017-01-01'), moment(),
  function onTick(tick) {
    console.log(moment(Number(tick[0])).format('YYYY-MM-DD HH:MM'), tick[1]);
  });
