import numpy as np
import pandas as pd
import QuantLib as ql
import yfinance as yf
from dataclasses import dataclass
from scipy.optimize import least_squares
from typing import List, Tuple

@dataclass
class NelsonSiegelParams:
    beta0: float
    beta1: float
    beta2: float
    lam: float

def nelson_siegel(tenor: float, p: NelsonSiegelParams) -> float:
    """
    Standard Nelson–Siegel factor model at tenor (years).
    """
    tau = max(1e-8, tenor)
    x = (1.0 - np.exp(-p.lam * tau)) / (p.lam * tau)
    return p.beta0 + p.beta1 * x + p.beta2 * (x - np.exp(-p.lam * tau))

@dataclass
class DividendNSModel:
    growth_params: NelsonSiegelParams   # g(T)
    premia_params: NelsonSiegelParams   # theta(T)

    def g(self, tenor: float) -> float:
        return nelson_siegel(tenor, self.growth_params)

    def theta(self, tenor: float) -> float:
        return nelson_siegel(tenor, self.premia_params)

# ---------- Data ----------

def get_dividends_data(ticker: str, start_date: str = "2005-01-01", n_years: int = 5) -> Tuple[pd.Series, pd.Series, float, float]:
    """
    Pull dividend data for given ticker starting from some past date to now.

    Args
    ----
    ticker : str
        Stock ticker symbol.
    start_date : str (default = "2005-01-01")
        Date from which to start collecting historical dividend data ('YYYY-MM-DD').
    n_years : int (default = 5)
        Number of years to aggregate when computing the average recent log dividend growth.
        
    Returns
    -------
    yearly_div_sums : pd.Series
        Historical cash dividends as yearly totals.
    log_div_growth : pd.Series
        Realized log dividend growth.
    avg_growth : float
        Average historical log dividend growth since input date.
    recent_growth : float
        Average recent log dividend growth.
    """
    
    # Create ticker object
    tk = yf.Ticker(ticker)

    # Historical cash dividends
    div = tk.dividends
    div = div[div.index >= start_date]

    # Aggregate to yearly totals
    yearly_div_sums = div.groupby(div.index.year).sum()
    yearly_div_sums.name = "dividend"

    # Realized log dividend growth
    log_div_growth = np.log(yearly_div_sums / yearly_div_sums.shift(1)).dropna()
    log_div_growth.name = "log_div_growth"

    # Long-run & recent average growth
    avg_growth = log_div_growth.mean()
    if len(log_div_growth) >= n_years:
        recent_growth = log_div_growth.tail(n_years).mean()
    else:
        recent_growth = avg_growth

    return yearly_div_sums, log_div_growth, avg_growth, recent_growth

# ---------- Fit Growth Parameters ----------

def implied_pd_ratio_from_ns(g_params: NelsonSiegelParams, premia_params: NelsonSiegelParams, 
                             risk_free_ts: ql.YieldTermStructureHandle, max_horizon_years: int = 30) -> float: # today_ql: ql.Date, cal_ql: ql.Calendar, day_counter_ql: ql.DayCounter, 
    """
    Compute model-implied price–dividend ratio from Nelson–Siegel growth & risk-premium curves
    and the risk-free curve, on a yearly grid up to max_horizon_years.

    Args
    ----
    g_params: NelsonSiegelParams 
        Nelson–Siegel growth curve parameters: beta0, beta1, beta2, lam.
    premia_params: NelsonSiegelParams
        Nelson–Siegel dividend risk-premium curve parameters: beta0, beta1, beta2, lam.
    risk_free_ts: ql.YieldTermStructureHandle
        Risk-free curve.
    max_horizon_years: int (default = 30)
        Maximum horizon (in calendar years) for the dividend yield curve.

    Returns
    -------
    pd_ratio : float
        The model-implied price–dividend ratio.
    """
    # Simple yearly grid: T = 1, 2, ..., N
    N = max_horizon_years
    pd_ratio = 0.0
    cumulative = 0.0

    for n in range(1, N + 1):
        T = float(n)

        # Risk-free zero rate
        y_T = risk_free_ts.zeroRate(T, ql.Continuous, ql.Annual).rate()

        # Expected dividend growth & risk premium at horizon T
        g_T = nelson_siegel(T, g_params)
        theta_T = nelson_siegel(T, premia_params)

        # Increment cumulative exponent for year n
        # (g - y + theta) is the "discounted risk-adjusted dividend growth"
        cumulative += g_T - y_T + theta_T

        # Contribution of year n dividend to S/D0
        pd_ratio += np.exp(cumulative)

    return float(pd_ratio)

def fit_ns_growth(avg_growth: float, recent_growth: float, pd_obs: float, risk_free_ts: ql.YieldTermStructureHandle,
                  premia_params: NelsonSiegelParams, growth_params_seed: List[float] = [0.01, -0.02, 0.01, 0.7], 
                  growth_params_lb: List[float] = [-0.1, -0.3, -0.3, 0.01], growth_params_ub: List[float] = [0.2, 0.3, 0.3, 5.0],
                  w_matching: Tuple[float, float, float] = (2.0, 1.0, 1.0), max_horizon_years: int = 30) -> NelsonSiegelParams:
    """
    Fit Nelson–Siegel growth term-structure parameters g(T) using least-squares by matching:
      - model-implied price–dividend ratio to 'pd_obs',
      - long-run average growth to historical 'avg_growth', and 
      - short-run (last 5y) average growth to 'recent_growth'.
    
    Args
    ----
    avg_growth : float
        Average historical log dividend growth since input date.
    recent_growth : float
        Average recent log dividend growth.
    pd_obs : float
        Observed price–dividend ratio using last full-year dividend.
    risk_free_ts : ql.YieldTermStructureHandle
        Risk-free curve.
    premia_params : NelsonSiegelParams
        Nelson–Siegel dividend risk-premium curve parameters: beta0, beta1, beta2, lam.
    growth_params_seed : List[float] (default = [0.01, -0.02, 0.01, 0.7]) 
        Initial seeds for Nelson–Siegel growth curve parameters as a vector: [beta0, beta1, beta2, lambda].
    growth_params_lb : List[float] (default = [-0.10, -0.30, -0.30, 0.01])
        Lower bound Nelson–Siegel growth curve parameter vector: [beta0, beta1, beta2, lambda].
    growth_params_ub : List[float] (default = [0.20, 0.30, 0.30, 5.00])
        Upper bound Nelson–Siegel growth curve parameter vector: [beta0, beta1, beta2, lambda].
    w_matching : Tuple[float, float, float] (default = (2.0, 1.0, 1.0))
        Residual weights to emphasize the relative importance of price–dividend match vs growth match.
        Ordered as [w_pd, w_long, w_short].
    max_horizon_years : int (default = 30)
        Maximum horizon (in calendar years) for the dividend yield curve.

    Returns
    -------
    fitted : NelsonSiegelParams
        Fitted parameters for the Nelson–Siegel growth curve.
    """
    w_pd, w_long, w_short = w_matching
    horizons = np.arange(1, max_horizon_years + 1, dtype = float)  # 1..N in years

    def residuals(p):
        
        g_params = NelsonSiegelParams(*p)

        # Price-dividend ratio match
        pd_model = implied_pd_ratio_from_ns(g_params, premia_params, risk_free_ts, max_horizon_years)
        e_pd = np.log(pd_model) - np.log(pd_obs)

        # Long-run average growth match
        g_vals = np.array([nelson_siegel(T, g_params) for T in horizons])
        g_long_model = g_vals.mean()
        e_long = g_long_model - avg_growth

        # Short-run average growth match
        short_mask = horizons <= 5.0
        if short_mask.sum() > 0:
            g_short_model = g_vals[short_mask].mean()
        else:
            g_short_model = g_long_model
        e_short = g_short_model - recent_growth

        return np.array([w_pd * e_pd, w_long * e_long, w_short * e_short])

    res = least_squares(residuals, x0 = np.array(growth_params_seed, dtype = float), 
                        bounds = (growth_params_lb, growth_params_ub), method = "trf")
    fitted = NelsonSiegelParams(*res.x)
    return fitted

# ---------- Build Dividend Yield Term Structure ----------

def build_dividend_yield_curve(today_ql: ql.Date, calendar_ql: ql.Calendar, day_counter_ql: ql.DayCounter,
                               risk_free_ts: ql.YieldTermStructureHandle, ns_div_model: DividendNSModel, 
                               max_years: int = 30, pillars_per_year: int = 2) -> ql.YieldTermStructureHandle:
    """
    Build a dividend yield term structure using Nelson–Siegel models for expected dividend growth & dividend risk premia. 
    The methodology follows the approach used in the NBIM 2021 report 'Modelling Equity Market Term Structures'.

    For each future tenor T, the continuous-compounded dividend yield q(T) is computed as:
        
        q(T) = y(T) - g(T) + theta(T)

    where:
        - y(T) is the risk-free zero rate at maturity T,
        - g(T) is the Nelson–Siegel dividend growth term structure,
        - theta(T) is the Nelson–Siegel dividend risk premia term structure.

    A set of pillar dates is generated out to 'max_years', with 'pillars_per_year' subdivisions per calendar year. 
    For each pillar maturity the implied dividend yield is computed and used to construct a YieldTermStructureHandle.

    Args
    ----
    today_ql : ql.Date 
        Evaluation date for the dividend term structure.
    calendar_ql : ql.Calendar 
        QuantLib calendar used to advance dates when generating maturity pillars.
    day_counter_ql : ql.DayCounter 
        Day-count convention used to compute year fractions between 'today_ql' & each pillar maturity.
    risk_free_ts : ql.YieldTermStructureHandle 
        Risk-free term structure used to obtain y(T) at each tenor T.
    ns_div_model : DividendNSModel 
        Object containing two Nelson–Siegel term structures:
            - growth_params: g(T), expected dividend growth term structure
            - premia_params: theta(T), dividend risk premium term structure
    max_years : int (default = 30) 
        Maximum horizon (in calendar years) for the dividend yield curve.
    pillars_per_year : int (default = 2)
        Number of term-structure pillars to create per year (e.g., 2 = semiannual, 4 = quarterly).

    Returns
    -------
    div_ts_handle: ql.YieldTermStructureHandle
        Dividend yield curve derived from dividend growth and risk premium term structures.
    """
    # By convention, spot dividend yield q(0) = 0
    zero_div_yields = [0.0]
    dates = [today_ql]

    # Build pillar tenors
    num_steps = max_years * pillars_per_year
    for k in range(1, num_steps + 1):
        
        t = k / pillars_per_year # tenor in years
        years = int(np.floor(t))
        months_fraction = t - years
        months = int(round(months_fraction * 12))
        mat_date = calendar_ql.advance(today_ql, ql.Period(years, ql.Years))
        if months > 0:
            mat_date = calendar_ql.advance(mat_date, ql.Period(months, ql.Months))
        dates.append(mat_date)

        # Year fraction from today
        T = day_counter_ql.yearFraction(today_ql, mat_date)

        # Risk-free zero rate at T (continuous comp, annual)
        y_T = risk_free_ts.zeroRate(T, ql.Continuous, ql.Annual).rate()

        # Expected dividend growth and risk premium
        g_T = ns_div_model.g(T)
        theta_T = ns_div_model.theta(T)

        # Implied dividend yield (continuous comp)
        q_T = y_T - g_T + theta_T
        zero_div_yields.append(q_T)

    div_zero_curve = ql.ZeroCurve(dates, zero_div_yields, day_counter_ql, calendar_ql)
    div_ts_handle = ql.YieldTermStructureHandle(div_zero_curve)
    return div_ts_handle

def compute_div_yield_from_ts(df: pd.DataFrame, dividend_ts: ql.YieldTermStructureHandle, 
                              day_counter_ql: ql.DayCounter) -> pd.DataFrame:
    """
    For each quote in the options chain, map its expiry (as a pandas Timestamp object) to a 
    continuous dividend yield q(T) based on the estimated dividend term structure.

    Args
    ----
    df : pd.DataFrame
        Option chains data. Must contain 'expiry' column.
    
    dividend_ts: ql.YieldTermStructureHandle
        QuantLib dividend yield term structure derived from dividend growth and risk premium term structures.

    day_counter_ql: ql.DayCounter
        Day-count convention used to determine time to expiry.

    Returns
    -------
    df : pd.DataFrame
        Option chains data containing one new column:
            - 'div_yield': Continuous dividend yield q(T), specific to each expiry.
    """ 
    if 'div_yield' in df.columns:
        df = df.rename(columns = {'div_yield': 'div_yield_proxy'})
    
    def div_yield_from_ts(expiry_ts: pd.Timestamp) -> float:
        """ Maps an option's expiry to a continuous dividend yield q(T) based on dividend term structure."""
        expiry_ql = ql.Date(expiry_ts.day, expiry_ts.month, expiry_ts.year)
        q = dividend_ts.zeroRate(expiry_ql, day_counter_ql, ql.Continuous, ql.Annual).rate() # continuous comp, annual frequency
        return float(q)
    
    # Compute q(T) for each unique expiry
    unique_expiries = df['expiry'].dropna().unique()
    q_by_expiry = {}
    for exp in unique_expiries:
        q_by_expiry[exp] = div_yield_from_ts(pd.Timestamp(exp))
    
    df['div_yield'] = df['expiry'].map(q_by_expiry)
    return df

