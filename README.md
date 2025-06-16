# MT5 Trading Bot

This project contains an automated trading bot for MetaTrader 5 that uses machine learning models.

## Configuration

When run for the first time, the bot creates a `config.json` file with the following default values:

```json
{
  "interval": 1,
  "max_ops": 5,
  "min_confidence": 0.75,
  "backtest_days": 182,
  "auto": false,
  "login": null,
  "server": null,
  "alpha": 0.7
}
```

### `alpha`
Controls the weight of the supervised model when combining probabilities with the unsupervised model inside the trading loop. The value must be between `0` and `1` and defaults to `0.7`.
