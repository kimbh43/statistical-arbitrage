# -*- coding: utf-8 -*-
import numpy as np
from quant.helpers import *
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from pykalman import KalmanFilter

def compute_spread(series_independent, series_dependent, gamma, mu, name=None):
    """
    Compute the spread of a portfolio given independent and dependent series.

    Parameters:
    series_independent (np.ndarray or pd.Series): Independent series data.
    series_dependent (np.ndarray or pd.Series): Dependent series data.
    gamma (float or np.ndarray): Gamma parameter(s), array or scalar.
    mu (float or np.ndarray): Mu parameter(s), array or scalar.

    Returns:
    np.ndarray or pd.Series: The computed weights.
    np.ndarray or pd.Series: The computed spread.
    """
    # Ensure x, y, and gamma are numpy arrays for broadcasting
    x = np.asarray(series_independent)
    y = np.asarray(series_dependent)
    gamma = np.asarray(gamma)
    
    # Check if gamma is a scalar or a 1-dimensional array
    if np.isscalar(gamma):
        w_spread = np.array([1, -gamma]) / (1 + gamma)
    elif gamma.ndim == 1:
        if series_independent.shape[0] != gamma.size:
            raise ValueError("gamma array must have the same number of elements as X.")
        w_spread = np.column_stack((np.ones_like(gamma), -gamma)) / (1 + gamma)[:, np.newaxis]
    else:
        raise ValueError("gamma must be either a scalar or a 1-dimensional array.")
    
    # Compute the spread
    spread = w_spread[:, 0] * y + w_spread[:, 1] * x - mu / (1 + gamma)
    spread = pd.Series(spread, index=series_independent.index)

    return w_spread, spread

def generate_signal(spread, threshold_long, threshold_short, pct_training = 0.7):
    """
    Generate a trading signal based on the Z-score and given thresholds for long and short positions.

    Parameters:
    - spread (np.ndarray or pd.Series): The computed spread.
    - threshold_long: The threshold value to enter long positions
    - threshold_short: The threshold value to enter short positions
    
    Returns:
    - An array of trading signals where 1 represents a long position, -1 represents a short position, and 0 represents no position
    """   
    # Calculate the number of observations for training
    T = len(spread)
    T_trn = round(pct_training * T)
    
    # Calculate mean and variance of the normalized_spread up to the T_trn (training period)
    spread_mean = np.nanmean(spread[:T_trn])
    spread_var = np.nanvar(spread[:T_trn])

    # Compute the Z-score
    z_score = (spread - spread_mean) / np.sqrt(spread_var)

    # Initialize the signal array and threshold array
    signal = np.zeros(len(z_score))
    threshold_long = np.full_like(z_score, threshold_long)
    threshold_short = np.full_like(z_score, threshold_short)
    
    # Initial position
    signal[0] = 0
    if z_score.iloc[0] <= threshold_long[0]:
        signal[0] = 1
    elif z_score.iloc[0] >= threshold_short[0]:
        signal[0] = -1
    
    # Loop through Z_score array
    for t in range(1, len(z_score)):
        if signal[t-1] == 0:  # if we were in no position
            if z_score.iloc[t] <= threshold_long[t]:
                signal[t] = 1
            elif z_score.iloc[t] >= threshold_short[t]:
                signal[t] = -1
        elif signal[t-1] == 1:  # if we were in a long position
            if z_score.iloc[t] >= 0:
                signal[t] = 0
            else:
                signal[t] = 1
        else:  # if we were in a short position
            if z_score.iloc[t] <= 0:
                signal[t] = 0
            else:
                signal[t] = -1
    
    T_trn = len(signal) * 2 // 3    
    plt.figure(figsize=(10, 5))
    plt.plot(z_score, label='Z-score')
    plt.step(range(len(signal)), signal, where='post', label='Trading Signal')
    plt.axvline(x=T_trn, color='blue', lw=2, label='End of Training Period')
    plt.title("Z-score and Trading Signal")
    ax = plt.gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.xticks(np.arange(0, len(z_score), 30))
    plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
    plt.tight_layout()  # Adjust layout to prevent clipping of tick-labels
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.legend(loc="upper left")
    plt.show()
    
    return z_score, signal

def linear_regression(series_independent, series_dependent):
    """
    Performs a simple linear regression on two series representing independent and dependent variables.

    The function takes two arguments which are expected to be pandas Series or similar types that
    can be converted to numpy arrays. It checks if the inputs are indeed numpy arrays and then
    proceeds to reshape the independent variable for compatibility with sklearn's LinearRegression class.

    It fits a linear model to the data and returns the intercept and coefficient of the independent
    variable, effectively giving the slope of the fitted line.

    Args:
    - series_independent (Series): A pandas Series representing the independent variable (predictor).
    - series_dependent (Series): A pandas Series representing the dependent variable (response).

    Returns:
    - tuple: A tuple containing the intercept and coefficient (slope) of the regression line.

    Raises:
    - ValueError: If the inputs are not numpy arrays or cannot be converted to numpy arrays.
    """
    # Ensure the input series are numpy arrays
    if not isinstance(series_independent.values, np.ndarray) or not isinstance(series_dependent.values, np.ndarray):
        raise ValueError("Both series_independent and series_dependent must be numpy arrays.")

    # Reshape the data. Independent variable needs to be in the format of
    # [[x1], [x2], [x3], ...] even if it's a single feature.
    X = series_independent.values.reshape(-1, 1)
    y = series_dependent.values

    # Perform linear regression using LinearRegression class
    lm_model = LinearRegression()
    lm_model.fit(X, y)  # Fit the model to the data

    # Return the intercept and slope of the regression line
    # lm_model.intercept_ is a scalar, while lm_model.coef_ is an array.
    # Since we are performing simple linear regression, we return lm_model.coef_[0] to get the slope value.
    return lm_model.intercept_, lm_model.coef_[0]

def estimate_mu_gamma_LS(series_independent, series_dependent, pct_training= 2/3):
    """
    Estimate mu and gamma parameters using least squares on two numpy arrays.
    
    Parameters:
    - series_independent (np.array): A numpy array representing the independent variable.
    - series_dependent (np.array): A numpy array representing the dependent variable.
    - pct_training (float): Percentage of the dataset to be used for training.
    
    Returns:
    - dict: A dictionary with estimated mu and gamma as numpy arrays.
    """
    
    # Calculate the number of observations for training
    T = len(series_dependent)
    T_trn = round(pct_training * T)
    
    # Perform least squares regression on the training data
    X_train = series_independent[:T_trn]
    y_train = series_dependent[:T_trn]
    mu, gamma = linear_regression(X_train, y_train)
    
    # Create numpy arrays for mu and gamma, replicating the coefficients for all time points
    mu = np.full(T, mu)
    gamma = np.full(T, gamma)
    
    return {'mu': mu, 'gamma': gamma}

def estimate_mu_gamma_rolling_LS(series_independent, series_dependent, lookback, shift, pct_training= 2/3):
    """
    Perform rolling window linear regression between two time series.
    
    Parameters:
    - series_independent (pd.Series): Time series data for the independent variable.
    - series_dependent (pd.Series): Time series data for the dependent variable.
    - lookback (int): The number of observations to use in each rolling window.
    - shift (int): The number of observations to shift the window by after each regression.
    - pct_training (float): Percentage of the dataset to be used for training.
    
    Returns:
    - dict: A dictionary with estimated mu and gamma as numpy arrays.
    """
    # Calculate the number of observations for training
    T = len(series_dependent)
    T_trn = round(pct_training * T)
    
    # Perform least squares regression on the training data
    X_train = series_independent[:T_trn]
    y_train = series_dependent[:T_trn]
    
    # Calculate rolling window start points
    t0_update = range(lookback, len(X_train) - shift, shift)
    mu_rolling_LS = pd.Series(np.nan, index=X_train.index)
    gamma_rolling_LS = pd.Series(np.nan, index=X_train.index)
    
    # Perform rolling linear regression
    for t0 in t0_update:
        # Extract the lookback period for the current window
        X = X_train.iloc[(t0-lookback):t0]
        y = y_train.iloc[(t0-lookback):t0]
        mu, gamma = linear_regression(X, y)
        # Update the rolling estimates with the current model's parameters
        mu_rolling_LS.iloc[t0] = mu  # intercept
        gamma_rolling_LS.iloc[t0] = gamma  # slope

    # Forward-fill the NaN values with the last valid estimate of coefficients
    mu_rolling_LS = mu_rolling_LS.ffill()
    gamma_rolling_LS = gamma_rolling_LS.ffill()

    # Backward-fill the NaN values with the next valid estimate of coefficients
    mu_rolling_LS = mu_rolling_LS.bfill()
    gamma_rolling_LS = gamma_rolling_LS.bfill()

    # Apply a moving average filter to smooth the series
    window_size = 30
    mu_rolling_LS = mu_rolling_LS.rolling(window=window_size, min_periods=1).mean()
    gamma_rolling_LS = gamma_rolling_LS.rolling(window=window_size, min_periods=1).mean()
    
    # Forward-fill any remaining NaN values after smoothing
    mu_rolling_LS = mu_rolling_LS.ffill()
    gamma_rolling_LS = gamma_rolling_LS.ffill()

    # Plotting the estimated 'mu' values over time
    # plt.figure(figsize=(12, 6))
    # plt.plot(mu_rolling_LS.index, mu_rolling_LS.values, label='Estimated mu', color='blue')
    # plt.title('Estimated mu over Time')
    # plt.xlabel('Time')
    # plt.ylabel('mu')
    # plt.legend()
    # plt.show()

    return {'mu': mu_rolling_LS, 'gamma': gamma_rolling_LS}

def estimate_mu_gamma_kalman_filter(series_independent, series_dependent, pct_training= 2/3):
    """
    Estimate mu and gamma parameters using a Kalman Filter on two time series.

    Parameters:
    - series_independent (np.ndarray): Time series data for the independent variable.
    - series_dependent (np.ndarray): Time series data for the dependent variable.
    - pct_training (float): Percentage of the dataset to be used for training.

    Returns:
    - dict: A dictionary with estimated mu and gamma as numpy arrays.    
    """

    # Calculate the number of observations for training
    T = len(series_dependent)
    T_trn = round(pct_training * T)
    
    # Perform least squares regression on the training data
    X_train = series_independent[:T_trn]
    y_train = series_dependent[:T_trn]
    
    # Constants for Kalman Filter
    state_cov_multiplier = np.power(0.01, 2) 
    observation_cov = 1.3e-7
    observation_matrix_stepwise = np.array([[X_train.iloc[0], 1]])
    observation_stepwise = y_train.iloc[0]

    # Initialize the Kalman Filter
    kf = KalmanFilter(n_dim_obs=1, n_dim_state=2,
                      initial_state_mean=np.ones(2),                      
                      initial_state_covariance=np.ones((2, 2)),           
                      transition_matrices=np.eye(2),                      
                      observation_matrices=observation_matrix_stepwise,   
                      observation_covariance=observation_cov,                           
                      transition_covariance= np.eye(2)*state_cov_multiplier)                   

    state_means_stepwise, state_covs_stepwise = kf.filter(observation_stepwise)             
    
    # Preallocate numpy arrays for performance
    n_timesteps = len(X_train)
    means_trace = np.zeros((n_timesteps, 2))
    covs_trace = np.zeros((n_timesteps, 2, 2))

    means_trace[0] = state_means_stepwise[0]
    covs_trace[0] = state_covs_stepwise[0]
    
    # Run the Kalman Filter through all timesteps
    for step in range(1, n_timesteps):
        observation_matrix_stepwise = np.array([[X_train.iloc[step], 1]])
        observation_stepwise = y_train.iloc[step]
    
        state_means_stepwise, state_covs_stepwise = kf.filter_update(
            means_trace[step - 1], covs_trace[step - 1],
            observation=observation_stepwise,
            observation_matrix=observation_matrix_stepwise)
    
        means_trace[step] = state_means_stepwise.data
        covs_trace[step] = state_covs_stepwise

    mu_kf = pd.Series(np.array(means_trace)[:,1], index=X_train.index)
    gamma_kf = pd.Series(np.array(means_trace)[:,0], index=X_train.index)
    
    # Apply a moving average filter to smooth the series
    window_size = 30
    mu_kf_smooth = mu_kf.rolling(window=window_size, min_periods=1).mean()
    gamma_kf_smooth = gamma_kf.rolling(window=window_size, min_periods=1).mean()
    
    # Forward-fill any remaining NaN values after smoothing
    mu_kf_smooth = mu_kf_smooth.ffill()
    gamma_kf_smooth = gamma_kf_smooth.ffill()
    
    # Extract and plot the estimated mu values over time
    # plt.plot(gamma_kf_smooth)
    # plt.title('Estimated mu over Time using Kalman Filter')
    # plt.xlabel('Time')
    # plt.ylabel('mu')
    # plt.show()
    
    return {'mu': mu_kf_smooth, 'gamma': gamma_kf_smooth}