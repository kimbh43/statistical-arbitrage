# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from itertools import combinations
import statsmodels.tsa.stattools as ts
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from quant.helpers import *

def calculate_normalized_distances(prices_df):
    '''
    Calculate the Euclidean distance for each unique pair of stocks in the dataframe
    and return a sorted dictionary of these distances in ascending order.
    
    Parameters:
    - prices_df (DataFrame): A pandas DataFrame where the index represents dates and each column
    represents the price series of a ticker.

    Returns:
    - sorted_distances (dict): A dictionary with keys as pairs of stock names and values as the
    Euclidean distance between them, sorted in ascending order.
    '''
    normalized_data = prices_df.apply(lambda x: x / x.iloc[0])
    distances = {}  # Dictionary to store the calculated distances

    # Calculate distances using combinations to avoid redundant pairs
    for s1, s2 in combinations(normalized_data.columns, 2):
        # Calculate Euclidean distance using np.linalg.norm for efficiency
        dist = np.linalg.norm(normalized_data[s1] - normalized_data[s2])
        distances[f'{s1}-{s2}'] = dist

    # Sort the dictionary by distance in ascending order
    sorted_distances = dict(sorted(distances.items(), key=lambda item: item[1]))

    return sorted_distances

def linear_regression_and_adf(series_independent, series_dependent):
    """
    Perform linear regression and ADF test on the residuals.

    Parameters:
    - series_independent: numpy array, independent variable for regression
    - series_dependent: numpy array, dependent variable for regression

    Returns:
    - p-value from the ADF test on the residuals
    - gamma from linear regression
    - mu from linear regression
    """
    mu, gamma = linear_regression(series_independent, series_dependent)
    # Calculate residuals
    X = series_independent.values.reshape(-1, 1)
    y = series_dependent.values

    yfit = gamma * X + mu
    residuals = y - yfit

    # Perform ADF test on residuals
    adf_result = ts.adfuller(residuals.flatten(), maxlag=1)
    p_value = adf_result[1]

    # Return p-value, coefficient (gamma), and intercept (mu)
    return p_value, mu, gamma

def cointegrated_ADF_test(series_1, series_2):
    """
    Test for cointegration between two time series using the Cointegrated Augmented Dickey-Fuller test
    on the residuals from two linear regressions.

    Parameters:
    - series_1: numpy array, first time series
    - series_2: numpy array, second time series

    Returns:
    - tuple: (gamma, regression_type)
      - gamma: float or False. It is the hedge ratio when cointegration is found (i.e., the coefficient 
        from the linear regression of the dependent variable on the independent variable), or False if 
        no cointegration is detected based on the p-value threshold.
      - regression_type: int. It indicates the direction of the cointegration relationship:
          - 0: No cointegration detected.
          - 1: 'series_1' is the independent variable and 'series_2' is the dependent variable.
          - 2: 'series_1' is the dependent variable and 'series_2' is the independent variable.
    """
    p_value_1, gamma_1, _ = linear_regression_and_adf(series_1, series_2)
    p_value_2, gamma_2, _ = linear_regression_and_adf(series_2, series_1)

    # Check if either p-value is less than the significance level
    if p_value_1 < 0.05 or p_value_2 < 0.05:
        # Return the gamma associated with the minimum p-value
        return (gamma_1, 1) if p_value_1 < p_value_2 else (gamma_2, 2)
    else:
        return (False, 0)

def johansen_test(series_1, series_2):
    """
    Compute the gamma_johansen value based on the Johansen cointegration test.

    Parameters:
    - series_1: numpy array, first time series
    - series_2: numpy array, second time series

    Returns:
    - gamma_johansen if both tests pass the 95% critical value, False otherwise
    """
    # Concatenate the Series from the dictionary into a DataFrame
    data = pd.concat([series_1, series_2], axis=1)

    # Perform the Johansen cointegration test
    jh_results = coint_johansen(data, det_order=0, k_ar_diff=1)
    
    # Extract the eigen-vectors
    v1 = jh_results.evec[:, 0]

    # Calculate gamma_johansen
    gamma_johansen = -v1[1] / v1[0]

    # Initialize results list
    results = []

    # Check the Trace statistic against the 95% critical values
    for idx in range(2):
        if jh_results.cvt[idx][1] < jh_results.lr1[idx]:  # 95% critical value index is 1
            results.append(True)
        else:
            results.append(False)

    # Check if both tests pass the 95% critical value
    if all(results):
        return gamma_johansen
    else:
        return False

def create_screening_result_df(filtered_data, original_data):
    """
    Creates a DataFrame with pairs of tickers that pass both cointegration tests.
    
    Parameters:
    filtered_data (pd.DataFrame): DataFrame containing the filtered data for the tickers.
    original_data (pd.DataFrame): DataFrame containing SP500 symbols and corresponding GICS Sectors.
    
    Returns:
    result_df (pd.DataFrame): DataFrame with the results containing pairs, distances, and sectors.
    """
    pairs_distances = calculate_normalized_distances(filtered_data)
    flattened_list_top20 = [[*k.split('-'), v] for k, v in list(pairs_distances.items())[:20]]
    
    # Create a dictionary to map tickers to their sectors
    sector_dict = pd.Series(original_data['GICS Sector'].values, index=original_data['Symbol']).to_dict()

    # Initialize an empty list to store the results
    result = []

    # Loop through each pair and perform the tests
    for pair_1, pair_2, distance in flattened_list_top20:
        if cointegrated_ADF_test(filtered_data[pair_1], filtered_data[pair_2]) and \
           johansen_test(filtered_data[pair_1], filtered_data[pair_2]):
            sector_1 = sector_dict.get(pair_1, "Sector Not Found")
            sector_2 = sector_dict.get(pair_2, "Sector Not Found")
            result.append([pair_1, pair_2, distance, sector_1, sector_2])

    # Convert the result list into a DataFrame
    result_df = pd.DataFrame(result, columns=['Pair 1', 'Pair 2', 'Distance', 'Sector 1', 'Sector 2'])

    return result_df