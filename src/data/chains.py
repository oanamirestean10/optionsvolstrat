import os
import numpy as np
import pandas as pd
import pandas_market_calendars as mcal
import QuantLib as ql
import yfinance as yf
from typing import List

# ---------- Pull Raw Option Chain Data ----------

def market_close_asof(date_local: pd.Timestamp, cal_code: str = "XNYS") -> List[pd.Timestamp]:
    """
    Return the most recent market-close timestamp in UTC following a given market calendar.

    Args
    ----
    date_local : pd.Timestamp
        Local timestamp for which we want to find the most recent market-close.
    
    cal_code : str
        Code corresponding to market calendar, e.g. XNYS (NYSE), XLON, XETR.
    
    Returns
    -------
    most_recent_close : pd.Timestamp
        The date of the most recent market-close as of date_local (tz-aware, UTC).
    
    prev_close : pd.Timestamp
        The date of the second most recent market-close as of date_local (tz-aware, UTC).
    """
    
    cal = mcal.get_calendar(cal_code)

    # Build a small date window around date of interest to locate prev/next sessions
    date_utc = date_local.tz_convert("UTC")
    start = (date_utc - pd.Timedelta(days = 10)).date()
    end = (date_utc + pd.Timedelta(days = 10)).date()

    # Full schedule with market_open/market_close in calendar's local timezone
    sched = cal.schedule(start_date = start, end_date = end)

    # Convert schedule times to UTC for robust comparisons
    sched_utc = sched.copy()
    sched_utc["market_open_utc"] = sched_utc["market_open"].dt.tz_convert("UTC")
    sched_utc["market_close_utc"] = sched_utc["market_close"].dt.tz_convert("UTC")
    past_closes = sched_utc[sched_utc["market_close_utc"] <= date_utc]["market_close_utc"]

    # Is the market open?
    date_str = date_utc.date().isoformat()
    date_row = sched_utc.loc[date_str] if date_str in sched_utc.index else None

    # Determine the most recent completed close
    if date_row is not None:
        use_today = (date_utc >= date_row["market_close_utc"])
        if use_today:
            most_recent_close = date_row["market_close_utc"]
            prev_close = past_closes.iloc[-2]
        else:
            most_recent_close = past_closes.iloc[-1]
            prev_close = past_closes.iloc[-2]
    else:
        most_recent_close = past_closes.iloc[-1]
        prev_close = past_closes.iloc[-2]
        
    return [most_recent_close, prev_close]

def continuous_rate_from_annual_percent(pct_yield: float) -> float:
    """Convert a quoted annualized percent yield to continuous compounding."""
    y = float(pct_yield) / 100.0
    return np.log(1.0 + max(y, 0.0))

def fetch_single_exp_option_chain(ticker: str, exp_chain: yf.ticker.Ticker, exp: str, df_rows: List[pd.DataFrame], 
                                  as_of_date: pd.Timestamp, S: float, r: float, q: float) -> List[pd.DataFrame]:
    """
    Aggregate option chain for the given ticker at a specified expiration.

    Args
    ----
    ticker : str
        Stock ticker symbol.

    exp_chain : yf.ticker.Ticker   
        Option chain object containing two DataFrames:
            - The first contians all the call option chain data for the specified expiry,
            - The second contians all the put option chain data for the specified expiry,
    
    exp : str
        Option expiry date as 'YYYY-MM-DD'.
    
    df_rows : List[pd.DataFrame]
        List of containing option chain DataFrames, each specific to a single expiry.
    
    as_of_date : pd.Timestamp
        Today's date.
    
    S : float
        Spot price.
    
    r : float
        Risk-free rate.
    
    q : float
        Dividend yield.

    Returns
    -------
    df_rows : List[pd.DataFrame]
        List of containing option chain DataFrames, each specific to a single expiry, 
        with the option chain DataFrame corresponding to the input expiry appended.
    """
    for option_type, option_df in [("call", exp_chain.calls), ("put", exp_chain.puts)]:
        
        if option_df is None or option_df.empty:
            continue
        
        tmp = option_df.copy()
        tmp["type"] = option_type
        tmp["symbol"] = ticker
        tmp["expiry"] = pd.to_datetime(exp, utc=True)
        tmp["date"] = as_of_date
        tmp["underlying"] = S
        tmp["rate"] = r
        tmp["div_yield"] = q

        # Parse lastTradeDate and compute quote age (hours)
        if "lastTradeDate" in tmp.columns:
            tmp["lastTradeDate"] = pd.to_datetime(tmp["lastTradeDate"], utc=True, errors="coerce")
            tmp["quote_age_hours"] = (tmp["date"] - tmp["lastTradeDate"]).dt.total_seconds().clip(lower = 0) / 3600.0
        else:
            tmp["lastTradeDate"] = pd.NaT
            tmp["quote_age_hours"] = np.nan

        tmp.rename(columns={"impliedVolatility": "iv_mkt",
                            "lastPrice": "last"}, inplace=True)
        cols = ["date", "expiry", "symbol", "contractSymbol", "type", "inTheMoney", "strike", "bid", "ask", "last", 
                "underlying", "iv_mkt", "rate", "div_yield", "volume", "openInterest", "lastTradeDate", "quote_age_hours"]
        tmp = tmp[[c for c in cols if c in tmp.columns]]
        df_rows.append(tmp)
    
    return df_rows

def fetch_option_chains(ticker: str, market_cal: str, as_of: str | None = None) -> pd.DataFrame: 
    """
    Fetch raw option chain for the given ticker across all expiries yfinance.

    Args
    ----
    ticker : str
        Stock ticker symbol.
    
    market_cal : str
        Code corresponding to market calendar, e.g. XNYS (NYSE), XNAS (NASDAQ), XLON, XETR.
    
    as_of : str | None (default = None)
        Evaluation date as 'YYYY-MM-DD'. 

    Returns
    -------
    df: pd.DataFrame
        Option chain across all expiries & option types for the input ticker.
    """
    # Market-close timestamp
    if as_of is None:
        # Most recent market-close as of today
        now_utc = pd.Timestamp.now(tz = "UTC")
        [as_of, prev_close] = market_close_asof(now_utc, market_cal)
    else:
        # Market-close on given date
        date_utc = pd.Timestamp(as_of).tz_localize("UTC")
        date_utc = date_utc.normalize() + pd.Timedelta(hours = 23, minutes = 59)
        [as_of, prev_close] = market_close_asof(date_utc, market_cal)
    
    # Create ticker object
    tk = yf.Ticker(ticker)

    # Available option expiry dates
    expiries = tk.options
    if len(expiries) == 0:
        raise RuntimeError(f"No option expiries available from yfinance for {ticker} right now.")

    # Pull 1 day of price history
    hist = tk.history(start = prev_close, end = as_of)
    if hist.empty:
        raise RuntimeError(f"No price history returned for {ticker}.")
    
    # Use latest close as spot
    S0 = float(hist["Close"].iloc[-1])

    # Risk-Free Proxy: 13-Week T-Bill
    try:
        irx = yf.Ticker("^IRX").history(period="5d")["Close"].dropna().iloc[-1]
        r_cont = continuous_rate_from_annual_percent(irx)
    except Exception:
        r_cont = 0.02 # fallback
    
    # Dividend Yield Proxy: trailing 365-day dividends / spot
    try:
        divs = tk.dividends
        recent = divs[divs.index >= (as_of - pd.Timedelta(days=365))]
        # Sum the last 365 days of dividends and divide by spot
        q_cont = float(recent.sum() / S0) if not recent.empty else 0.0 # NOT continuous q but proxy
    except Exception:
        q_cont = 0.0 # fallback
    
    # Aggregate option chains
    rows = []
    for exp in expiries:
        chain = tk.option_chain(exp)
        rows = fetch_single_exp_option_chain(ticker, chain, exp, rows, as_of, S0, r_cont, q_cont)    
    df = pd.concat(rows, ignore_index=True)

    # Compute mid price from bid/ask when both exist; otherwise fall back to last price
    df["mid"] = np.where(np.isfinite(df[["bid", "ask"]]).all(axis = 1), 0.5 * (df["bid"] + df["ask"]), df["last"])
    
    # Compute time to expiry in years using Actual/365
    df["T"] = (pd.to_datetime(df["expiry"]) - pd.to_datetime(df["date"])).dt.days.clip(lower = 0).astype(float) / 365.0

    return df

def save_raw_data(df: pd.DataFrame, ticker: str, outdir: str, run_date: str):
    """
    Save the option chain dataframe as CSV:
        {outdir}/{ticker}_{run_date}.csv
    
    Args
    ----
    df : pd.DataFrame     
        Raw option chain data for 'ticker' as of 'run_date'.
    
    ticker : str
        Stock ticker symbol.
    
    outdir : str
        Path to directory where the data is to be saved.
    
    run_date : str
        Today's date as "YYYYMMDD".
    """
    filename = f"{ticker}_{run_date}.csv"
    path = os.path.join(outdir, filename)
    df.to_csv(path, index = False)
    print(f"Saved {run_date} {ticker} option chains to {path}")

def option_chains(ticker: str, market_cal: str, n_exp: int, as_of: str | None = None) -> pd.DataFrame: 
    """
    Get option chain for given ticker.

    Args
    ----
    ticker : str
        Stock ticker symbol.
    
    market_cal : str
        Code corresponding to market calendar, e.g. XNYS (NYSE), XNAS (NASDAQ), XLON, XETR.
    
    n_exp : int
        Number of expiries to include.
    
    as_of : str | None (default = None)
        Evaluation date as 'YYYY-MM-DD'. 

    Returns
    -------
    df: pd.DataFrame
        Option chain across all expiries & option types for the input ticker.
    """
    # Market-close timestamp
    if as_of is None:
        # Most recent market-close as of today
        now_utc = pd.Timestamp.now(tz = "UTC")
        [as_of, prev_close] = market_close_asof(now_utc, market_cal)
    else:
        # Market-close on given date
        date_utc = pd.Timestamp(as_of).tz_localize("UTC")
        date_utc = date_utc.normalize() + pd.Timedelta(hours = 23, minutes = 59)
        [as_of, prev_close] = market_close_asof(date_utc, market_cal)
    # Create ticker object
    tk = yf.Ticker(ticker)
    # Available option expiry dates
    opts = tk.options
    if len(opts) == 0:
        raise RuntimeError(f"No option expiries available from yfinance for {ticker} right now.")
    # List of n expiries at least a week out
    time_to_expiry = []
    for expiry_date in opts:
        t = int(max(1, (pd.to_datetime(expiry_date, utc = True) - as_of).days))
        if t >= 7:
            time_to_expiry.append(expiry_date)
    expiries = time_to_expiry[:n_exp]
    # Pull 1 day of price history
    hist = tk.history(start = prev_close, end = as_of)
    if hist.empty:
        raise RuntimeError(f"No price history returned for {ticker}.")
    # Use latest close as spot
    S0 = float(hist["Close"].iloc[-1])
    # Risk-Free Proxy: 13-Week T-Bill
    try:
        irx = yf.Ticker("^IRX").history(period="5d")["Close"].dropna().iloc[-1]
        r_cont = continuous_rate_from_annual_percent(irx)
    except Exception:
        r_cont = 0.02
    # Dividend Yield Proxy: trailing 365-day dividends / spot
    try:
        divs = tk.dividends
        recent = divs[divs.index >= (as_of - pd.Timedelta(days=365))]
        # Sum the last 365 days of dividends and divide by spot
        q_cont = float(recent.sum() / S0) if not recent.empty else 0.0 # NOT continuous q but proxy for
    except Exception:
        q_cont = 0.0
    # Aggregate option chains
    rows = []
    for exp in expiries:
        chain = tk.option_chain(exp)
        rows = fetch_single_exp_option_chain(ticker, chain, exp, rows, as_of, S0, r_cont, q_cont)    
    df = pd.concat(rows, ignore_index=True)
    # Compute mid price from bid/ask when both exist; otherwise fall back to last price
    df["mid"] = np.where(np.isfinite(df[["bid", "ask"]]).all(axis = 1), 0.5 * (df["bid"] + df["ask"]), df["last"])
    # Compute time to expiry in years using Actual/365
    df["T"] = (pd.to_datetime(df["expiry"]) - pd.to_datetime(df["date"])).dt.days.clip(lower = 0).astype(float) / 365.0
    return df

# ---------- Clean Data ----------

def subset_expiries(df: pd.DataFrame, num_exp: int, min_days_to_exp: int = 7) -> pd.DataFrame:
    """
    Subset option chains data based on expiry dates.
    
    Args
    ----
    df : pd.DataFrame
        Option chains data. Must contain 'date', 'expiry'.

    num_exp : int
        Number of expiries to keep in the subset.
    
    min_days_to_exp : int  
        Minimum number of days to the closest expiry.

    Returns
    -------
    df : pd.DataFrame 
        Option chains data subsetted to contain only the relevant expiries.
    """
    
    as_of = pd.to_datetime(df["date"].max()).tz_convert("UTC")
    all_expiries = df["expiry"].unique().tolist()
    expiries_subset = []
    
    for expiry in all_expiries:
        t_days = int(max(1, (pd.to_datetime(expiry, utc = True) - as_of).days))
        if t_days >= min_days_to_exp and len(expiries_subset) < num_exp:
            expiries_subset.append(expiry)
        if len(expiries_subset) == num_exp:
            break
    
    first_expiry = expiries_subset[0]
    last_expiry = expiries_subset[-1]
    df = df[(df["expiry"] >= first_expiry) & (df["expiry"] <= last_expiry)]
    return df

def clean_data(df: pd.DataFrame, tick: float = 0.01, min_oi: int = 10, age_hours: int = 48) -> pd.DataFrame:
    """
    Clean data and remove any invalid or illiquid quotes.

    Args
    ----
    df : pd.DataFrame
        Option chains data. 
        Must contain 'mid', 'strike', 'T', 'iv_mkt', 'bid', 'ask', 'volume', 'openInterest', 'quote_age_hours'.
    
    tick : float (default = 0.01)
        Minimum tick size. Default for equity options is usually 0.01.
    
    min_oi : int (default = 10)
        Minimum open interest to identify which quotes are illiquid.
    
    age_hours : int (default = 48)
        Threshold for the number of hours since last trade to mark a quote stale.
    
    Returns
    -------
    df : pd.DataFrame 
        Option chains data containing only valid and liquid quotes.
    """
    # Remove rows with non-positive prices/strikes or zero time
    df = df[(df["mid"] > 0) & (df["strike"] > 0) & (df["T"] > 0)]

    # Remove rows with missing IVs, bid or ask quotes
    df = df.dropna(subset = ["iv_mkt", "bid", "ask"])

    # Remove rows with obviously broken IVs: negative/zero, or absurdly high (> 500%)
    df.loc[df["iv_mkt"] <= 0, "iv_mkt"] = np.nan
    df.loc[df["iv_mkt"] > 5.0, "iv_mkt"] = np.nan
    df = df.dropna(subset = ["iv_mkt"]).reset_index(drop = True)

    # Identify invalid quotes: ask < bid, negative bid/ask, or zero book
    df["invalid_quote"] = ((df["ask"] < df["bid"]) | 
                           (df["bid"] < 0) | (df["ask"] <= 0) | 
                           ((df["bid"] == 0) & (df["ask"] == 0)))

    # Identify locked quotes with no liquidity: bid = ask & volume = 0 & very low OI
    df["locked_no_liquidity"] = ((df["ask"] - df["bid"] <= tick + 1e-12) & 
                                 (df["volume"].fillna(0) == 0) & 
                                 (df["openInterest"].fillna(0) < min_oi))
    
    # Identify stale quotes ("too old") that were last traded over age_hours since as_of
    df["stale_age"] = ((df["quote_age_hours"] > age_hours))

    # Drop these quotes
    df["drop"] = df[["invalid_quote", "locked_no_liquidity", "stale_age"]].any(axis=1)
    df = df[~df["drop"]]
    df = df.drop(columns=["invalid_quote", "locked_no_liquidity", "stale_age", "drop"])

    return df

def most_liquid_quotes(df: pd.DataFrame, spread_width: float = 0.25) -> pd.DataFrame:
    """
    Subset data to keep only the most liquid part of the option chains (strikes within +/-50% of forward)
    and quotes with reasonable bid–ask spreads.
    
    Args
    ----
    df : pd.DataFrame
        Option chains data. 
        Must contain 'underlying', 'rate', 'div_yield', 'T', 'strike', 'ask', 'bid', 'mid'.
    
    spread_width : float (default = 0.25)
        Width factor to maintain a reasonable bid–ask spread.
    
    Returns
    -------
    df : pd.DataFrame 
        Option chains data containing only the most liquid quotes.
    """
    # Forward price
    df["F"] = df["underlying"] * np.exp((df["rate"] - df["div_yield"]) * df["T"])
    
    # Keep only strikes within +/-50% of forward
    df = df[(df["strike"] >= 0.5 * df["F"]) & (df["strike"] <= 1.5 * df["F"])]
    
    # Reasonable bid–ask spread
    df = df[((df["ask"] - df["bid"]) / df["mid"]) < spread_width]
    return df

def compute_trading_days_to_exp(row: pd.Series) -> int:
    """
    Compute the number of NYSE trading days between the option quote date and its expiry.
    
    Args
    ----
    row : pd.Series
        A row of the option chains DataFrame representing an available option. 
        Must contain columns 'date', 'T_days'.

    Returns
    -------
    n_trading_days : int
        The number of trading days between the quote date & the expiry date for the option of interest.
    """
    date_row = pd.to_datetime(row["date"]).tz_convert("UTC")
    today_row = ql.Date(date_row.day, date_row.month, date_row.year)
    T_days_row = int(row["T_days"])
    expiry_row = today_row + T_days_row
    cal = ql.UnitedStates(ql.UnitedStates.NYSE)
    n_trading_days = cal.businessDaysBetween(today_row, expiry_row)
    return n_trading_days

def compute_expiry_liquidity(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute liquidity metrics and a a composite liquidity score for each expiry in the option chain.

    Args
    ----
    df : pd.DataFrame
        Option chain data with corresponding model-implied price & implied volatility estimates.
        Must contain columns 'expiry', 'ask', 'bid', 'mid'.
    
    Returns
    -------
    norm_liq_by_exp : pd.DataFrame
        DataFrame indexed by expiry and sorted by 'liquidity_score', with the most liquid expiries 
        at the top and the least liquid at the bottom. 
        Contains 4 columns:
            - 'total_volume_norm': Total volume by expiry, normalized,
            - 'total_oi_norm': Total open interest by expiry, normalized,
            - 'n_quotes_norm': Total number of quotes per expiry, normalized,
            - 'liquidity_score': Composite liquidity score.
    """

    df_liq = df.copy()
    df_liq["rel_spread"] = (df_liq["ask"] - df_liq["bid"]) / df_liq["mid"]

    liq_stats_by_exp = df_liq.groupby("expiry").agg(
        total_volume = ("volume", "sum"),
        total_oi = ("openInterest", "sum"),
        n_quotes = ("mid", "count"),
    )

    # Z-score normalization
    norm_liq_by_exp = (liq_stats_by_exp - liq_stats_by_exp.mean()) / liq_stats_by_exp.std()
    norm_liq_by_exp = norm_liq_by_exp.rename(columns = {
        'total_volume'  : 'total_volume_norm', 
        'total_oi'      : 'total_oi_norm', 
        'n_quotes'      : 'n_quotes_norm', 
    })

    # Composite score: higher = more liquid
    norm_liq_by_exp["liquidity_score"] = ( norm_liq_by_exp["total_volume_norm"] 
                                          + norm_liq_by_exp["total_oi_norm"] 
                                          + norm_liq_by_exp["n_quotes_norm"])

    return norm_liq_by_exp.sort_values("liquidity_score", ascending=False)

def select_most_liquid_expiries(df: pd.DataFrame, n_exp: int = 5) -> pd.DataFrame:
    """
    Subset the full option chain to only contain the n_exp most-liquid expiries.

    Args
    ----
    df : pd.DataFrame
        Option chain data with corresponding model-implied price & implied volatility estimates.
        Must contain columns 'expiry', 'ask', 'bid', 'mid'.
    
    n_exp : int (default = 5)
        Number of expiries to include in the option chain subset.

    Returns
    -------
    df_sub : pd.DataFrame
        Option chain subset that only contains the n_exp most-liquid expiries.
    """
    liq_ranked_by_expiry = compute_expiry_liquidity(df)
    best_expiries = liq_ranked_by_expiry.head(n_exp).index.tolist()
    df_sub = df[df["expiry"].isin(best_expiries)].copy()
    return df_sub
