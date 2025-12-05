import numpy as np
import pandas as pd
import QuantLib as ql
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from numpy.linalg import inv
from scipy.optimize import least_squares
from scipy.stats import norm
from typing import Dict, List, Optional, Tuple

# ---------- Data ----------

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

# ---------- Fit Simple Heston Model ----------

def build_vanilla_instruments(df: pd.DataFrame, today: ql.Date) -> List[ql.VanillaOption]:
    """
    Create VanillaOption objects for each option in the chains data.

    Args
    ----
    df : pd.DataFrame  
        Option chains data. Must contain 'expiry', 'date', 'type', 'strike'.
    
    today : ql.Date
        Evaluation date.

    Returns
    -------
    options : List[ql.VanillaOption]
        List of ql.VanillaOption objects corresponding to options available in chains.
    """
    options = []
    
    for _, df_row in df.iterrows():
        
        # Maturity
        T_days = int(max(1, (pd.to_datetime(df_row["expiry"]) - pd.to_datetime(df_row["date"])).days))
        maturity = today + T_days  # or cal.advance(today, T_days, ql.Days)
        exercise = ql.EuropeanExercise(maturity)

        # Payoff
        payoff = ql.PlainVanillaPayoff(ql.Option.Call if str(df_row["type"]).lower().startswith('c') else ql.Option.Put, float(df_row["strike"]))
        
        # Option Specification
        opt = ql.VanillaOption(payoff, exercise)
        options.append(opt)
    
    return options

def compute_greeks_and_bs_price(df: pd.DataFrame, options: List[ql.VanillaOption], S_q: ql.QuoteHandle,
                  r_ts: ql.YieldTermStructureHandle, q_ts: ql.YieldTermStructureHandle) -> pd.DataFrame:
    """
    Compute the Black-Scholes-Merton option price & greeks (delta, gamma, vega, theta, rho, dividend rho) 
    for each quote in the options chain.
    
    Args
    ----
    df : pd.DataFrame  
        Option chains data.
    
    options : List[ql.VanillaOption]
        List of ql.VanillaOption objects corresponding to option chains.
    
    S_q : ql.QuoteHandle
        Spot quote.
    
    r_ts : ql.YieldTermStructureHandle
        Risk-free curve.
    
    q_ts : ql.YieldTermStructureHandle
        Dividend curve.
    
    Returns
    -------
    df : pd.DataFrame
        Option chains data such that each quote has its Black-Scholes-Merton option price & greeks, 
        containing seven new columns:
            - 'bs_price' 
            - 'delta'
            - 'gamma'
            - 'vega'
            - 'theta'
            - 'rho'
            - 'dividend_rho'
    """
    
    bs_prices = []
    deltas = []
    gammas = []
    vegas = []
    thetas = []
    rhos = []
    dividend_rhos = []

    df = df.reset_index()
    for i, df_row in df.iterrows():

        # Build Volatility Curve
        sigma = float(df_row["iv_mkt"])
        vol_curve = ql.BlackVolTermStructureHandle(ql.BlackConstantVol(0, ql.TARGET(), sigma, ql.Actual365Fixed()))

        # Build Black-Scholes-Merton process
        process = ql.BlackScholesMertonProcess(S_q, q_ts, r_ts, vol_curve)
        
        # Corresponding VanillaOption object
        option = options[i]

        # Black-Scholes analytic pricing engine
        engine = ql.AnalyticEuropeanEngine(process)
        option.setPricingEngine(engine)

        # Compute option price & greeks
        bs_prices.append(option.NPV())
        deltas.append(option.delta())
        gammas.append(option.gamma())
        vegas.append(option.vega())
        thetas.append(option.theta())
        rhos.append(option.rho())
        dividend_rhos.append(option.dividendRho())

    df["bs_price"] = bs_prices
    df["delta"] = deltas
    df["gamma"] = gammas
    df["vega"] = vegas
    df["theta"] = thetas
    df["rho"] = rhos
    df["dividend_rho"] = dividend_rhos
    return df

# Calibration Weighting

def inverse_variance_weighting(bid: float, ask: float, S: float, K: float, T: float, r: float, q: float, iv: float, vega: float,
                               min_tick: float = 0.01, min_half_spread_iv: float = 0.0025, min_vega: float = 1e-5) -> float:
    """
    Give more weight to options with tight bid-ask spreads (as they are more reliable) in model calibration. 
    Convert each bid–ask spread in price space into a half-spread in IV space & assign weights using:
    w = 1/(spread^2).

    Args
    ----
    bid : float                    
        Bid quote.
    
    ask : float
        Ask quote.
    
    S : float                 
        Spot price.
    
    K : float                 
        Strike price.
    
    T : float                   
        Time to expiration as a year fraction.
    
    r : float                   
        Risk-free rate.
    
    q : float                
        Dividend yield.
    
    iv : float                   
        Implied volatility.
    
    vega : float                  
        Vega of option.
    
    min_tick : float (default = 0.01)
        Minimum tick size for bid-ask spread.
    
    min_half_spread_iv : float (default = 0.0025)
        Floor for IV half-spread.
    
    min_vega : float (default = 1e-5)          
        Minimum Vega threshold.
    
    Returns
    -------
    iv_weight : float
        The inverse-variance weight corresponding to the specified input quote. 
    """
    # Validate
    if not np.isfinite(bid) or not np.isfinite(ask) or ask <= bid or T <= 0 or iv <= 0:
        return np.nan
    
    # Price half-spread with tick floor
    half_spread_price = 0.5 * max(ask - bid, min_tick)

    # Convert price half-spread to IV half-spread
    if vega < min_vega:
        return np.nan
    half_spread_iv = half_spread_price / vega
    half_spread_iv = max(half_spread_iv, min_half_spread_iv)
    
    # Inverse-variance weight
    iv_weight = 1.0 / (half_spread_iv ** 2)
    return iv_weight

def wing_damping_weighting(K: float, F: float, wing_lambda: float = 0.5) -> float:
    """
    Down-weight options far from forward to focus calibration on 'core' region where vol smile is anchored using:
    w = exp(-lambda * |k|).
    
    Args
    ----
    K : float              
        Strike price.
    
    F : float              
        Forward price.
    
    wing_lambda : float (default = 0.5)   
        Parameter identifying the strength of exponential damping.
    
    Returns
    -------
    wing_weight : float 
        The wing-damping weight corresponding to the specified input quote. 
    """
    # Log-moneyness
    k = np.log(np.maximum(1e-12, (K / np.maximum(1e-12, F))))
    # Wing damping: as price moves away from strike, weight decays exponentially
    wing_weight = np.exp(-wing_lambda * np.abs(k))
    return wing_weight

def calibration_weighting(df: pd.DataFrame, lambda_wing: float = 0.5) -> pd.DataFrame:
    """
    Compute weight for each option that indicates how 'trustworthy' the quote is and how much it should 
    influence the volatility surface model calibration. Weights are normalized per expiry so one 
    maturity doesn't overwhelm model fit.

    Args
    ----
    df : pd.DataFrame      
        Option chains data. 
        Must contain 'bid', 'ask', 'underlying', 'strike', 'T', 'rate', 'div_yield', 'iv_mkt', 'vega', 'F', 'expiry'.
    
    wing_lambda : float (default = 0.5)    
        Parameter identifying the rate of exponential decay in wing damping.

    Returns
    -------
    df : pd.DataFrame 
        Option chains data where each option has corresponding weights indicating how 'trustworthy' the quote is.
        Four new columns containing computed weights:
            - 'w_iv': Inverse-IV-variance weights,
            - 'w_wing': Wing-damping weights,
            - 'w': 'w_iv' * 'w_wing',
            - 'w_norm': Weights normalized by expiry.
    """
    w_iv_list = []
    w_wing_list = []
    for _, df_row in df.iterrows():
        
        bid_w = df_row["bid"]
        ask_w = df_row["ask"]
        S_w = df_row["underlying"]
        K_w = df_row["strike"]
        T_w = df_row["T"]
        r_w = df_row["rate"]
        q_w = df_row["div_yield"]
        iv_w = df_row["iv_mkt"]
        vega_w = df_row["vega"]
        F_w = df_row["F"]
        
        # Inverse-IV-variance weight
        iv_weight = inverse_variance_weighting(bid_w, ask_w, S_w, K_w, T_w, r_w, q_w, iv_w, vega_w)
        w_iv_list.append(iv_weight)

        # Wing-damping weight
        wd_weight = wing_damping_weighting(K_w, F_w, lambda_wing)
        w_wing_list.append(wd_weight)

    df["w_iv"] = w_iv_list
    df["w_wing"] = w_wing_list
    df["w"] = df["w_iv"] * df["w_wing"]
    df = df.dropna(subset = ["w_iv", "w_wing", "w"]).reset_index(drop = True)

    # Normalize per expiry
    df["w_norm"] = 0.0
    for exp, idx in df.groupby("expiry").groups.items():
        idx = list(idx)
        w_sum = df.loc[idx, "w"].sum()
        df.loc[idx, "w_norm"] = df.loc[idx, "w"] / (w_sum if w_sum > 0 else 1.0)
    
    return df

# Fit Model (Minimize Price RMSE)

def build_heston_engine(S_q: ql.QuoteHandle, r_ts: ql.YieldTermStructureHandle, q_ts: ql.YieldTermStructureHandle, 
                        kappa: float, theta: float, sigma: float, rho: float, v0: float) -> Tuple[ql.HestonModel, ql.AnalyticHestonEngine]:
    """
    Create a stochastic Heston model and pricing engine for asset & variance evolution under the risk-neutral 
    measure, given a set of parameters.

    Args
    ----
    S_q : ql.QuoteHandle
        Spot quote.
    
    r_ts : ql.YieldTermStructureHandle
        Risk-free curve.
    
    q_ts : ql.YieldTermStructureHandle
        Dividend yield curve.
    
    kappa : float
        Fitted Heston parameter representing the mean reversion speed of the process variance.
    
    theta : float
        Fitted Heston parameter representing the long-run variance level.
    
    sigma : float
        Fitted Heston parameter representing the volatility of the process variance (vol of vol).
    
    rho : float):                            
        Fitted Heston parameter representing the correlation between asset returns and variance shocks.
    
    v0 : float
        Fitted Heston parameter representing the initial variance.
        
    Returns
    -------
    Tuple[ql.HestonModel, ql.AnalyticHestonEngine]
    """
    process = ql.HestonProcess(r_ts, q_ts, S_q, float(v0), float(kappa), float(theta), float(sigma), float(rho))
    model   = ql.HestonModel(process)
    engine  = ql.AnalyticHestonEngine(model)
    return model, engine

def compute_params_ci(params_hat: np.ndarray, residual_vec: np.ndarray, jacobian: np.ndarray, 
                      alpha: int = 0.05) -> Tuple[Dict[str, Tuple[float, float]], np.ndarray]:
    """
    Compute asymptotic confidence intervals and standard errors for calibrated Heston parameters 
    using the nonlinear least-squares covariance approximation.

    Args
    ----
    params_hat : np.ndarray of shape (5,)
        Estimated Heston parameter vector: [kappa, theta, sigma, rho, v0].
    
    residual_vec : np.ndarray of shape (m,)
        Vector of weighted residuals at the optimum returned by the nonlinear least-squares estimator.
    
    jacobian : np.ndarray of shape (m, 5)
        Jacobian matrix of the residuals with respect to the fitted parameters, evaluated at the optimum. 
    
    alpha : int (default = 0.05)
        Significance level for confidence intervals. (1 - alpha) will be the confidence level.
    
    Returns
    -------
    ci_dict : Dict[str, Tuple[float, float]]
        Dictionary mapping parameter names to their upper & lower confidence interval bounds
        for the specified confidence level alpha. Each value is a tuple: (lower_bound, upper_bound).
        The dictionary has the following keys: "kappa", "theta", "sigma", "rho", "v0".
    
    std_err : np.ndarray of shape (5,)
        Array of length 5 containing the standard errors of the estimates, derived from the 
        nonlinear least-squares covariance approximation.
    """
    m = residual_vec.size   # number of data points
    num_params = params_hat.size

    # Residual variance
    sigma2 = (residual_vec @ residual_vec) / max(m - num_params, 1)

    # Covariance matrix of parameters
    cov = sigma2 * inv(jacobian.T @ jacobian)

    # Standard errors
    std_err = np.sqrt(np.diag(cov))

    # (1 - alpha) CI
    z = norm.ppf(1 - alpha / 2)
    ci_lower = params_hat - z * std_err
    ci_upper = params_hat + z * std_err

    param_names = ["kappa", "theta", "sigma", "rho", "v0"]
    ci = {
        name: [float(lo), float(hi)]
        for name, lo, hi in zip(param_names, ci_lower, ci_upper)
    }

    return ci, std_err

def is_near_bounds(params: np.ndarray, lb: np.ndarray, ub: np.ndarray, tol_frac: float = 0.05) -> bool:
    """
    Determine if any fitted Heston parameter lies within 'tol_frac' of its lower or upper bound.

    Args
    ----
    params : np.ndarray  of shape (5,)
        Fitted Heston parameter vector: [kappa, theta, sigma, rho, v0].
    
    lb : np.ndarray  of shape (5,)
        Lower bounds for Heston parameters: kappa, theta, sigma, rho, v0.
    
    ub : np.ndarray of shape (5,)
        Upper bounds for Heston parameters: kappa, theta, sigma, rho, v0.
    
    tol_frac : float (default = 0.05)
        Fraction of the parameter range used to define the "near boundary" region.
        If a fitted parameter is within this fraction of its lower or upper bound,
        the solution is considered too close to the bounds.
    
    Returns
    -------
    bool
        True if any fitted Heston parameter lies within 'tol_frac' of its lower or upper bound.
    """
    lb = np.asarray(lb, dtype = float)
    ub = np.asarray(ub, dtype = float)
    params = np.asarray(params, dtype = float)

    width = ub - lb
    # absolute distance to bounds
    dist_lower = params - lb
    dist_upper = ub - params

    # threshold = 5% of range by default
    thresh = tol_frac * width

    near_lower = dist_lower <= thresh
    near_upper = dist_upper <= thresh

    return bool(np.any(near_lower | near_upper))

def fit_heston_ls(df: pd.DataFrame, options: List[ql.VanillaOption], S_q: ql.QuoteHandle, 
                  r_ts: ql.YieldTermStructureHandle, q_ts: ql.YieldTermStructureHandle, x0_seeds: List[float], 
                  lb: List[float], ub: List[float], alpha: float = 0.05, max_restarts: int = 5, 
                  bound_tol_frac: float = 0.05) -> Tuple[np.ndarray, Dict[str, Tuple[float, float]], np.ndarray]:
    """
    Calibrates Heston model to option chains by finding the parameter vector that minimizes the weighted 
    sum of squared price errors via non-linear least squares.

    If the fitted parameters lie within 'bound_tol_frac' of any lower/upper bound, the calibration is 
    repeated with a perturbed initial seed, up to 'max_restarts' times.
    
    Args
    ----
    df : pd.DataFrame                      
        Option chains data. Must contain columns 'mid' and 'w_norm'.
    
    options : List[ql.VanillaOption]
        List of ql.VanillaOption objects corresponding to input option chains. 
    
    S_q : ql.QuoteHandle
        Spot quote.
    
    r_ts : ql.YieldTermStructureHandle)
        Risk-free curve.
    
    q_ts : ql.YieldTermStructureHandle
        Dividend curve.
    
    x0_seeds : List[float]
        Initial seeds for Heston parameter fitting: kappa, theta, sigma, rho, v0.
    
    lb : List[float]
        Lower bounds for Heston parameters: kappa, theta, sigma, rho, v0.
    
    ub : List[float]                       
        Upper bounds for Heston parameters: kappa, theta, sigma, rho, v0.
    
    alpha : int (default = 0.05)
        Significance level for confidence intervals. (1 - alpha) will be the confidence level.
    
    max_restarts : int (default = 5)
        Maximum number of calibration attempts with different seeds.
    
    bound_tol_frac : float (default = 0.05)
            Fraction of the parameter range used to define the "near boundary" region.
            If a fitted parameter is within this fraction of its lower or upper bound,
            the solution is considered too close to the bounds.
    
    Returns
    -------
    params_hat : np.ndarray of shape (5,)
        Estimated Heston parameter vector: [kappa, theta, sigma, rho, v0].
    
    ci_dict : Dict[str, Tuple[float, float]]
        Dictionary mapping parameter names to their upper & lower confidence interval bounds
        for the specified confidence level alpha.
        Keys: 'kappa', 'theta', 'sigma', 'rho', 'v0'.
        Values: (lower_bound, upper_bound).
    
    std_err : np.ndarray of shape (5,)
        Array of length 5 containing the standard errors of the estimates, derived from the 
        nonlinear least-squares covariance approximation.
    """
    
    prices_mkt = df["mid"].to_numpy(float)
    weights = df["w_norm"].to_numpy(float)
    lb_arr = np.array(lb, dtype=float)
    ub_arr = np.array(ub, dtype=float)
    x0_base = np.array(x0_seeds, dtype=float)

    def resid(params: np.ndarray) -> np.ndarray:
        kappa, theta, sigma, rho, v0 = params
        _, engine = build_heston_engine(S_q, r_ts, q_ts, kappa, theta, sigma, rho, v0)

        model_prices = np.empty_like(prices_mkt)
        for i, opt in enumerate(options):
            opt.setPricingEngine(engine)
            model_prices[i] = float(opt.NPV())
        
        # Signed weighted residuals
        return np.sqrt(weights) * (prices_mkt - model_prices)

    # Try multiple restarts if fitted parameters are too close to bounds
    best_cost = np.inf
    best_params = None
    best_ci = None
    best_se = None
    rng = np.random.default_rng(42)

    for attempt in range(max_restarts + 1):

        if attempt == 0:
            x0 = x0_base.copy()
        else:
            # Random perturbation around base seed within ~20% of the parameter range
            width = ub_arr - lb_arr
            perturb = 0.2 * width * rng.normal(size = width.shape)
            x0 = np.clip(x0_base + perturb, lb_arr, ub_arr)

        res = least_squares(resid, x0, bounds = (lb_arr, ub_arr), xtol = 1e-8, ftol = 1e-8, gtol = 1e-8, max_nfev = 400)
        params_hat = res.x
        resid_vec = res.fun
        jac = res.jac

        # Compute CIs & standard errors for this attempt
        ci_dict, std_err = compute_params_ci(params_hat, resid_vec, jac, alpha)

        # Check if this solution is far enough from the bounds
        if not is_near_bounds(params_hat, lb_arr, ub_arr, tol_frac=bound_tol_frac):
            best_params = params_hat
            best_ci = ci_dict
            best_se = std_err
            break

        # If still near bounds, keep lowest cost as fallback
        if res.cost < best_cost:
            best_cost = res.cost
            best_params = params_hat
            best_ci = ci_dict
            best_se = std_err

    # Final outputs: either interior solution or best attempt
    return best_params, best_ci, best_se

# Compute Model Price & Implied Volatility

def compute_model_prices_and_ivs(df: pd.DataFrame, engine: ql.AnalyticHestonEngine, today: ql.Date, S_q: ql.QuoteHandle, 
                                 r_ts: ql.YieldTermStructureHandle, q_ts: ql.YieldTermStructureHandle) -> pd.DataFrame:
    """
    Compute Heston-derived prices and implied volatilities for input option chains.

    Args
    ----
    df : pd.DataFrame 
        Option chains data. Must contain 'expiry', 'date', 'type', 'strike'.
    
    engine : ql.AnalyticHestonEngine
        Calibrated Heston pricing engine.
    
    today : ql.Date
        Evaluation date.
    
    S_q : ql.QuoteHandle
        Spot quote.
    
    r_ts : ql.YieldTermStructureHandle
        Risk-free curve.
    
    q_ts : ql.YieldTermStructureHandle     
        Dividend yield curve.
        
    Returns
    -------
    df : pd.DataFrame
        Options chains data with two additional columns:
            - 'price_model': Heston-derived option prices,
            - 'iv_model': model-derived implied volatilities.
    """

    # Build a flat BSM volatility curve with arbitrary seed (doesn't matter)
    vol_seed = ql.BlackConstantVol(today, ql.NullCalendar(), 0.20, ql.Actual365Fixed())
    vol_ts_seed = ql.BlackVolTermStructureHandle(vol_seed)
    bsm_process = ql.BlackScholesMertonProcess(S_q, q_ts, r_ts, vol_ts_seed)

    bs_ivs = []
    model_prices = []
    for _, df_row in df.iterrows():
        
        # Maturity
        T_days = int(max(1, (pd.to_datetime(df_row["expiry"]) - pd.to_datetime(df_row["date"])).days))
        maturity = today + T_days
        exercise = ql.EuropeanExercise(maturity)

        # Create Vanilla option object
        option_type = ql.Option.Call if str(df_row["type"]).lower().startswith('c') else ql.Option.Put
        option = ql.VanillaOption(ql.PlainVanillaPayoff(option_type, float(df_row["strike"])), exercise)

        # Price option using calibrated Heston model
        option.setPricingEngine(engine)
        price_model = float(option.NPV())
        model_prices.append(price_model)

        # Compute implied volatility via Black-Scholes-Merton
        iv = option.impliedVolatility(price_model, bsm_process, 1e-8, 500, 1e-6, 10.0)
        bs_iv = float(iv)
        bs_ivs.append(bs_iv)
    
    df["price_model"] = model_prices
    df["iv_model"] = bs_ivs
    return df

def compute_residuals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the weighted and unweighted residuals for the fitted model prices and implied volatilities.
    
    Args
    ----
    df : pd.DataFrame                 
        Option chains data with fitted model prices and implied volatilities.
        Must contain 'mid', 'price_model', 'iv_mkt', 'iv_model', 'w_norm'.
    
    Returns
    -------
    df : pd.DataFrame
        Options chains data with four additional columns:
            - 'price_resid': Unweighted price residuals,
            - 'iv_resid': Unweighted implied volatility residuals,
            - 'price_resid_weighted': Weighted price residuals,
            - 'iv_resid_weighted': Weighted implied volatility residuals,
    """
    
    weights = df["w_norm"].to_numpy(float)
    prices_mkt = df["mid"].to_numpy(float)
    prices_model = df["price_model"].to_numpy(float)
    iv_mkt = df["iv_mkt"].to_numpy(float)
    iv_model = df["iv_model"].to_numpy(float)

    # Unweighted residuals
    price_residuals = prices_mkt - prices_model
    iv_residuals = iv_mkt - iv_model
    df["price_resid"] = price_residuals
    df["iv_resid"] = iv_residuals

    # Weighted residuals
    weighted_price_residuals = np.sqrt(weights) * price_residuals
    weighted_iv_residuals = np.sqrt(weights) * iv_residuals
    df["price_resid_weighted"] = weighted_price_residuals
    df["iv_resid_weighted"] = weighted_iv_residuals
    
    return df

def compute_price_rmse(df: pd.DataFrame) -> float:
    """
    Evaluate the root mean square error between market price and model price.
    
    Args
    ----
    df : pd.DataFrame                 
        Option chains data. Must contain 'price_model', 'mid'.
    
    Returns
    -------
    rmse : float
        RMSE between market price and model price.
    """
    model_option_prices = df["price_model"].to_numpy(float)
    market_option_prices = df["mid"].to_numpy(float)
    rmse = float(np.sqrt(np.nanmean((model_option_prices - market_option_prices)**2)))
    return rmse

# ---------- MC Simulated Volatility Paths ----------

def simulate_heston_paths(process: ql.HestonProcess, maturity: float, n_steps: int, n_paths: int, 
                          seed: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate price & instantaneous variance paths based on the stochastic dynamics of (S_t, v_t) 
    from the calibrated Heston process using Monte Carlo simulations. 
    
    NOTE:   QuantLib uses Quadratic Exponential discretization for the variance process and 
            log-Euler discretization for the asset price.
    
    Args
    ----
    process : ql.HestonProcess     
        Calibrated Heston process describing the stochastic dynamics of (S_t, v_t).
    
    maturity : float               
        The simulation time horizon T expressed as a year fraction (e.g. 0.25 = 3 months).
        If starting from calendar dates, convert with a day counter: T = daycounter.yearFraction(today, expiry_date).
    
    n_steps : int                  
        Number of time steps between 0 and T in the Monte Carlo grid (dt = maturity / n_steps).
    
    n_paths : int                  
        Number of independent Monte Carlo sample paths to generate.
    
    seed : int (default = 42)                     
        Random seed for QuantLib's pseudo-random sequence generator, ensuring reproducibility of the simulated paths.

    Returns
    -------
    t_grid : np.ndarray, shape (n_steps + 1,)
        Simulation grid time steps from 0 to T, as year fractions.
    
    S_paths_sim : np.ndarray, shape (n_paths, n_steps + 1)
        Simulated price paths of the underlying asset S_t.
    
    v_paths_sim : np.ndarray, shape (n_paths, n_steps + 1)
        Simulated instantaneous variance paths v_t.
    """
    # Generate a multivariate Gaussian path consistent with the SDE dynamics for Heston process & time grid
    times = ql.TimeGrid(maturity, n_steps)
    # Gaussian sequence generator
    n_factors = process.factors() # 2 factors: S_t, v_t
    dimension = n_factors * n_steps
    uniform_seq_gen = ql.UniformRandomSequenceGenerator(dimension, ql.UniformRandomGenerator(seed))
    gaussian_seq_gen = ql.GaussianRandomSequenceGenerator(uniform_seq_gen)
    # Multi-path generator
    path_gen = ql.GaussianMultiPathGenerator(process, times, gaussian_seq_gen, False)
    S_paths_sim = np.zeros((n_paths, n_steps + 1))
    v_paths_sim = np.zeros((n_paths, n_steps + 1))
    for i in range(n_paths):
        sample = path_gen.next()
        multi_path = sample.value()
        S_paths_sim[i, :] = [multi_path[0][j] for j in range(len(times))]
        v_paths_sim[i, :] = [multi_path[1][j] for j in range(len(times))]
    t_grid = np.array([times[j] for j in range(len(times))])
    return t_grid, S_paths_sim, v_paths_sim

def est_mc_option_price(spot_prices_T: np.ndarray, strike: float, T_exercise: float, r_ts: ql.YieldTermStructureHandle, 
                        n_paths: int, opt_type: str, ci_level: float = 0.95) -> Tuple[float, float, float]:
    """
    Compute the option price estimate and confidence interval across the Monte Carlo simulated spot price paths at expiry.
    
    Args
    ----
    spot_prices_T : np.ndarray (n_paths,)            
        Simulation spot prices at maturity/exercise (T).
    
    strike : float        
        Option strike price.
    
    T_exercise : float         
        Time to maturity/exercise, as a year fraction.
    
    r_ts : ql.YieldTermStructureHandle
        Risk-free term structure.
    
    n_paths : int
        Number of Monte Carlo sample paths.
    
    opt_type : str (default = None)    
        Option type. Specify "c" for calls and "p" for puts.
    
    ci_level : float (default = 0.95)
        Confidence interval level, as a decimal (e.g. 95% = 0.95).
    
    Returns
    -------
    mc_mean : float
        Option price estimate at maturity/exercise.
    
    ci_lower : float
        Lower bound of the confidence interval for the option price estimate at maturity/exercise.
    
    ci_upper : float
        Upper bound of the confidence interval for the option price estimate at maturity/exercise.
    """
    payoffs = np.maximum(spot_prices_T - strike, 0.0) if opt_type.lower() == "c" else np.maximum(strike - spot_prices_T, 0.0)
    discount_factor = r_ts.discount(T_exercise)
    discounted_payoffs = discount_factor * payoffs
    mc_mean = discounted_payoffs.mean()
    mc_std  = discounted_payoffs.std(ddof = 1)
    z = norm.ppf(1 - ci_level / 2)
    ci_lower = float(mc_mean - z * mc_std / np.sqrt(n_paths))
    ci_upper = float(mc_mean + z * mc_std / np.sqrt(n_paths))
    return mc_mean, ci_lower, ci_upper

def est_mc_option_prices(df: pd.DataFrame, spot_prices_T: np.ndarray, T_exercise: float, r_ts: ql.YieldTermStructureHandle, 
                        n_paths: int, ci_level: float = 0.95) -> pd.DataFrame:
    """
    Compute the option price estimates & corresponding confidence intervals based on the Monte Carlo simulated 
    spot and volatility paths for all options in the input chain.

    Args
    ----
    df : pd.DataFrame
        Option chains data. Must contain columns 'strike' and 'type'.
    
    spot_prices_T : np.ndarray (n_paths,)            
        Simulation spot prices at maturity (T).
    
    T_exercise : float         
        Time to maturity/exercise, as a year fraction.
    
    r_ts : ql.YieldTermStructureHandle
        Risk-free term structure.
    
    n_paths : int
        Number of Monte Carlo sample paths.
    
    ci_level : float (default = 0.95)
        Confidence interval level, as a decimal (e.g. 95% = 0.95).
    
    Returns
    -------
    df : pd.DataFrame
        Option chains data with option price estimates and confidence intervals for each quote based on 
        the Monte Carlo simulated spot and volatility paths. 
            - mc_price: Monte Carlo option price estimates,
            - ci_lower: Confidence interval lower bound for the Monte Carlo option price estimates,
            - ci_upper: Confidence interval upper bound for the Monte Carlo option price estimates.
    """
    price_estimates = []
    price_ci_lower = []
    price_ci_upper = []
    for _, option_row in df.iterrows():
        K = float(option_row["strike"])
        option_type = option_row["type"]
        price_est, est_ci_lower, est_ci_upper = est_mc_option_price(spot_prices_T, K, T_exercise, r_ts, 
                                                                    n_paths, option_type[0], ci_level)
        price_estimates.append(price_est)
        price_ci_lower.append(est_ci_lower)
        price_ci_upper.append(est_ci_upper)
    df["mc_price"] = price_estimates
    df["ci_lower"] = price_ci_lower
    df["ci_upper"] = price_ci_upper
    return df

# ---------- Plotting ----------

# Calibrated Heston

def plot_model_prices(df: pd.DataFrame, ticker: str, n_exp: int, option_type: Optional[str] = None, expiry: Optional[str] = None):
    """
    Plot model price vs market price for each strike and the corresponding price residuals.
    
    Args
    ----
    df : pd.DataFrame
        Option chains data with corresponding model-implied price & implied volatility estimates.
        Must contain columns 'strike', 'mid', 'price_model', 'price_resid'.
    
    ticker : str
        Stock ticker symbol.
    
    n_exp : int
        Number of expiries included in the options chain.
    
    option_type : Optional[str] (default = None)    
        Option type of data to be plotted. Default value is None, indicating both calls & puts. 
        Specify "c" for calls only and "p" for puts only.
    
    expiry : Optional[str] (default = None)
        If the data contains only a single expiry, specify this date as a string. 
    """

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(20, 8))

    # Model Price vs Market Price
    ax[0].scatter(df['strike'], df['mid'], label = 'Market', color = 'c', alpha = 0.5, s = 50)
    ax[0].scatter(df['strike'], df['price_model'], label = 'Heston', color = 'm', alpha = 0.5, s = 30)
    ax[0].set_xlabel("Strike")
    ax[0].set_ylabel("Price")
    ax[0].set_title(f"Option Price: Model vs Market")
    ax[0].legend()

    # Residuals
    ax[1].scatter(df['strike'], df['price_resid'], color = 'r')
    ax[1].set_xlabel("Strike")
    ax[1].set_ylabel(" Market Price - Model Price")
    ax[1].set_title(f"Price Residuals")

    # Option Type
    if option_type is not None: 
        option_type_str = "Calls" if option_type.lower() == "c" else "Puts"
    else:
        option_type_str = "Options"
    
    # Figure Title
    if expiry is not None:
        expiry_datetime = datetime.strptime(expiry, "%Y-%m-%d %H:%M:%S%z")
        expiry_date_str = expiry_datetime.date().strftime("%Y-%m-%d")
        fig_title = f"{ticker} {option_type_str}, {expiry_date_str} Expiry"
    else:
        fig_title = f"{ticker} {option_type_str} Across {n_exp} Expiries"
    
    fig.suptitle(fig_title, fontsize=18)
    plt.show()

def plot_model_iv(df: pd.DataFrame, ticker: str, n_exp: int, option_type: Optional[str] = None, expiry: Optional[str] = None):
    """
    Plot model IV vs market IV for each strike and the corresponding IV residuals.
    
    Args
    ----
    df : pd.DataFrame
        Option chains data with corresponding model-implied price & implied volatility estimates.
        Must contain columns 'strike', 'iv_mkt', 'iv_model', 'iv_resid'.
    
    ticker : str
        Stock ticker symbol.
    
    n_exp : int
        Number of expiries included in the options chain.
    
    option_type : Optional[str] (default = None)    
        Option type of data to be plotted. Default value is None, indicating both calls & puts. 
        Specify "c" for calls only and "p" for puts only.
    
    expiry : Optional[str] (default = None)
        If the data contains only a single expiry, specify this date as a string. 
    """

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(20, 8))

    # Model IV vs Market IV
    ax[0].scatter(df['strike'], df['iv_mkt'], label = 'Market', color = 'c', alpha = 0.5, s = 50)
    ax[0].scatter(df['strike'], df['iv_model'], label = 'Heston', color = 'm', alpha = 0.5, s = 30)
    ax[0].set_xlabel("Strike")
    ax[0].set_ylabel("Implied Volatility")
    ax[0].set_title(f"Implied Volatility: Model vs Market")
    ax[0].legend()

    # Residuals
    ax[1].scatter(df['strike'], df['iv_resid'], color = 'r')
    ax[1].set_xlabel("Strike")
    ax[1].set_ylabel("Market IV - Model IV")
    ax[1].set_title(f"Implied Volatility Residuals")

    # Option Type
    if option_type is not None: 
        option_type_str = "Calls" if option_type.lower() == "c" else "Puts"
    else:
        option_type_str = "Options"
    
    # Figure Title
    if expiry is not None:
        expiry_datetime = datetime.strptime(expiry, "%Y-%m-%d %H:%M:%S%z")
        expiry_date_str = expiry_datetime.date().strftime("%Y-%m-%d")
        fig_title = f"{ticker} {option_type_str}, {expiry_date_str} Expiry"
    else:
        fig_title = f"{ticker} {option_type_str} Across {n_exp} Expiries"
    
    fig.suptitle(fig_title, fontsize=18)
    plt.show()

# MC Simulated Volatility Paths

def plot_sim_paths(t_grid: np.ndarray, spot_paths: np.ndarray, var_paths: np.ndarray, 
                   spot_quantiles: Tuple[np.ndarray, np.ndarray, np.ndarray], 
                   var_quantiles: Tuple[np.ndarray, np.ndarray, np.ndarray],
                   ticker: str, expiry: str, option_type: Optional[str] = None):
    """
    Plot the first 50 spot price paths & variance paths simulated under today’s calibrated 
    risk-neutral Heston dynamics for options expiring at the same specified 'expiry'.
    
    Args
    ----
    t_grid : np.ndarray            
        Simulation grid time steps from 0 to T, as year fractions.
    
    spot_paths : np.ndarray        
        Simulated spot price paths of the underlying asset.
    
    var_paths : np.ndarray         
        Simulated instantaneous variance paths.
    
    spot_quantiles : Tuple[np.ndarray, np.ndarray, np.ndarray]
        5%, 50%, 95% quantile bands for the simulated Heston spot paths.
    
    var_quantiles : Tuple[np.ndarray, np.ndarray, np.ndarray]
        5%, 50%, 95% quantile bands for the simulated Heston variance paths.
    
    ticker : str                   
        Underlying ticker symbol.
    
    expiry : str                  
        Option expiry date as a string.
    
    option_type : Optional[str] (default = None)    
        Option type of data to be plotted. Specify "c" for calls and "p" for puts. None indicates both calls & puts.
    """
    spot_p05, spot_p50, spot_p95 = spot_quantiles
    var_p05, var_p50, var_p95 = var_quantiles

    fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize = (20, 8))

    # Spot Paths
    for i in range(50):
        ax[0].plot(t_grid, spot_paths[i, :], alpha = 0.4)
    ax[0].plot(t_grid, spot_p50, color = "red", label = "Median")
    ax[0].fill_between(t_grid, spot_p05, spot_p95, color = "grey", alpha = 0.25, label = "5–95% Percentile")
    ax[0].set_xlabel("Time (years)")
    ax[0].set_ylabel("Spot Price")
    ax[0].set_title("Simulated Heston Spot Paths")
    ax[0].legend()

    # Variance Paths
    for i in range(50):
        ax[1].plot(t_grid, var_paths[i, :], alpha = 0.4)
    ax[1].plot(t_grid, var_p50, color = "red", label = "Median")
    ax[1].fill_between(t_grid, var_p05, var_p95, color = "grey", alpha = 0.25, label = "5–95% Percentile")
    ax[1].set_xlabel("Time (years)")
    ax[1].set_ylabel("Variance")
    ax[1].set_title("Simulated Variance Paths")
    ax[1].legend()
    
    # Figure Title
    if option_type is not None: 
        option_type_str = "Calls" if option_type.lower() == "c" else "Puts"
    else:
        option_type_str = "Options"
    expiry_datetime = datetime.strptime(expiry, "%Y-%m-%d %H:%M:%S%z")
    expiry_date_str = expiry_datetime.date().strftime("%Y-%m-%d")
    fig_title = f"{ticker} {option_type_str}, {expiry_date_str} Expiry"
    
    fig.suptitle(fig_title, fontsize=18)
    plt.show()

def plot_distr_at_maturity(spot_T: np.ndarray, vol_T: np.ndarray, current_spot: float, 
                           ticker: str, expiry: str, option_type: Optional[str] = None):
    """
    For a set of options with the same expiry, plot the distributions of the Monte Carlo 
    simulated future spot prices & instantaneous volatilities at maturity for the 
    underlying asset.
    
    Args
    ----
    spot_T : np.ndarray
        Simulated future spot prices of the underlying asset at maturity T.     
    
    vol_T : np.ndarray        
        Simulated instantaneous volatilities of the underlying asset at maturity T.
    
    current_spot : float         
        The current spot price of the underlying asset, as of today.
    
    ticker : str                   
        Underlying ticker symbol.
    
    expiry : str                  
        Option expiry date as a string.
    
    option_type : Optional[str] (default = None)    
        Option type of data to be plotted. Specify "c" for calls and "p" for puts. 
        None indicates both calls & puts.
    """
    
    # Future spot prices at maturity: Mean & CI
    spot_T_mean = spot_T.mean()
    spot_T_p05  = np.percentile(spot_T, 5)
    spot_T_p95  = np.percentile(spot_T, 95)

    # Instantaneous vol at maturity: Mean & CI
    vol_T_mean = vol_T.mean()
    vol_T_p05  = np.percentile(vol_T, 5)
    vol_T_p95  = np.percentile(vol_T, 95)

    hls_palette = sns.color_palette("hls", 8)
    husl_palette = sns.color_palette("husl", 8)

    fig, (ax1, ax2) = plt.subplots(nrows = 1, ncols = 2, figsize = (20, 8))

    # Distribution of Future Spot Prices at Maturity
    kde_spot_T = sns.kdeplot(spot_T, alpha = 0, ax = ax1).lines[0]
    spot_prices_kde, spot_denisty_kde = kde_spot_T.get_data()
    spot_prices_ci = spot_prices_kde[(spot_prices_kde >= spot_T_p05) & (spot_prices_kde <= spot_T_p95)]
    spot_denisty_ci = spot_denisty_kde[(spot_prices_kde >= spot_T_p05) & (spot_prices_kde <= spot_T_p95)]
    ax1.vlines(x = [spot_T_p05, spot_T_p95], ymin = 0, ymax = [spot_denisty_ci[0], spot_denisty_ci[-1]], 
               colors = husl_palette[6], linestyles = 'dotted', lw = 2)
    # Histogram & KDE
    sns.kdeplot(spot_T, fill = True, color = husl_palette[5], alpha = 0.25, ec = hls_palette[5], ax = ax1)
    sns.histplot(spot_T, bins = 50, stat = "density", color = hls_palette[4], alpha = 0.75, edgecolor = husl_palette[4], ax = ax1)
    sns.kdeplot(spot_T, color = hls_palette[5], lw = 2, ax = ax1, label = "KDE")
    # 95% Confidence Interval
    ax1.plot(spot_prices_ci, spot_denisty_ci, color = husl_palette[6], linewidth = 2, label = "95% CI")
    # Mean Future Spot Price
    ax1.vlines(x = [spot_T_mean], ymin = 0, ymax = [spot_denisty_kde[(spot_prices_kde >= spot_T_mean)][0]], 
            colors = husl_palette[7], lw = 2, label = "Mean Future Spot")
    # Current Spot Price
    ax1.vlines(x = [current_spot], ymin = 0, ymax = [spot_denisty_kde[(spot_prices_kde >= current_spot)][0]], 
            colors = husl_palette[3], lw = 2, label = "Current Spot")
    ax1.set_xlabel("Spot Price")
    ax1.set_ylabel("Density")
    ax1.set_title("Distribution of Future Spot Prices at Maturity")
    ax1.legend()
    
    # Distribution of Instantaneous Vol at Maturity
    kde_vol_T = sns.kdeplot(vol_T, alpha = 0, ax = ax2).lines[0]
    vol_kde, vol_denisty_kde = kde_vol_T.get_data()
    vol_ci = vol_kde[(vol_kde >= vol_T_p05) & (vol_kde <= vol_T_p95)]
    vol_denisty_ci = vol_denisty_kde[(vol_kde >= vol_T_p05) & (vol_kde <= vol_T_p95)]
    ax2.vlines(x = [vol_T_p05, vol_T_p95], ymin = 0, ymax = [vol_denisty_ci[0], vol_denisty_ci[-1]], 
               colors = husl_palette[6], linestyles = 'dotted', lw = 2)
    # Histogram & KDE
    sns.kdeplot(vol_T, fill = True, color = husl_palette[5], alpha = 0.25, ec = hls_palette[5], ax = ax2)
    sns.histplot(vol_T, bins = 50, stat = "density", color = hls_palette[4], alpha = 0.75, edgecolor = husl_palette[4], ax = ax2)
    sns.kdeplot(vol_T, color = hls_palette[5], lw = 2, ax = ax2, label = "KDE")
    # 95% Confidence Interval
    ax2.plot(vol_ci, vol_denisty_ci, color = husl_palette[6], linewidth = 2, label = "95% CI")
    # Mean Instantaneous Vol
    ax2.vlines(x = [vol_T_mean], ymin = 0, ymax = [vol_denisty_ci[(vol_ci >= vol_T_mean)][0]], 
               colors = husl_palette[7], lw = 2, label = "Mean Vol")
    ax2.set_xlabel("Instantaneous Volatility")
    ax2.set_ylabel("Density")
    ax2.set_title("Distribution of Instantaneous Vol at Maturity")
    ax2.legend()

    # Figure Title
    if option_type is not None: 
        option_type_str = "Calls" if option_type.lower() == "c" else "Puts"
    else:
        option_type_str = "Options"
    expiry_datetime = datetime.strptime(expiry, "%Y-%m-%d %H:%M:%S%z")
    expiry_date_str = expiry_datetime.date().strftime("%Y-%m-%d")
    fig_title = f"{ticker} {option_type_str}, {expiry_date_str} Expiry"

    fig.suptitle(fig_title, fontsize=18)
    plt.show()

def plot_mc_est_prices(df: pd.DataFrame, ticker: str, expiry: str, option_type: Optional[str] = None):
    
    """
    Plot Monte Carlo option price estimate vs market price for each strike and the corresponding residuals.
    
    Args
    ----
    df : pd.DataFrame
        Option chains data with corresponding model-implied price & implied volatility estimates.
        Must contain columns 'strike', 'mid', 'price_model', 'price_resid'.
    
    ticker : str
        Stock ticker symbol.
    
    option_type : Optional[str] (default = None)    
        Option type of data to be plotted. Default value is None, indicating both calls & puts. 
        Specify "c" for calls only and "p" for puts only.
    
    expiry : Optional[str] (default = None)
        If the data contains only a single expiry, specify this date as a string. 
    """
    fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize = (20, 8))

    # Market Price vs MC Estimate
    ax[0].scatter(df['strike'], df['mid'], label = 'Market', color = 'c', alpha = 0.5, s = 50)
    ax[0].scatter(df['strike'], df['mc_price'], label = 'MC Estimate', color = 'm', alpha = 0.5, s = 30)
    # ax[0].fill_between(df['strike'], df['ci_lower'], df['ci_upper'], color = 'm', alpha = 0.15, label = "95% Confidence Interval")
    ax[0].set_xlabel("Strike")
    ax[0].set_ylabel("Price")
    ax[0].set_title(f"Option Price: Market vs Monte Carlo Estimate")
    ax[0].legend()

    # Residuals
    residuals = df['mid'] - df['mc_price']
    ax[1].scatter(df['strike'], residuals)
    ax[1].set_xlabel("Strike")
    ax[1].set_ylabel("Market Price - Model Price")
    ax[1].set_title(f"Residuals")

    # Figure Title
    if option_type is not None: 
        option_type_str = "Calls" if option_type.lower() == "c" else "Puts"
    else:
        option_type_str = "Options"
    expiry_datetime = datetime.strptime(expiry, "%Y-%m-%d %H:%M:%S%z")
    expiry_date_str = expiry_datetime.date().strftime("%Y-%m-%d")
    fig_title = f"{ticker} {option_type_str}, {expiry_date_str} Expiry"

    fig.suptitle(fig_title, fontsize = 18)
    plt.show()
