# optionsvolstrat

Repo Structure:

/data/          # raw & processed (parquet)
/src/
  data/        # loaders, cleaning, corporate actions, earnings
  features/    # RV/IV features, term/skew, events
  models/      # ARIMA/GARCH/HAR + ML + ensembles
  signals/     # IV forecasts -> trade signals
  execution/   # cost & liquidity models, hedging
  backtest/    # engine, portfolio, risk
  eval/        # metrics, DM/SPA, attribution
/notebooks/    # EDA, model dev, result reports
/conf/         # universe, filters, costs
/reports/      # auto-generated HTML/markdown

