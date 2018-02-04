const store = require('./store.js');

var data = new store.Data(store.Ticket.BTCUSDT);

data.get(store.Interval.D);
