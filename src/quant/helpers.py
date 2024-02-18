import yfinance 
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

def get_sp500_table():
    """
    Fetches the table of S&P 500 companies from the Wikipedia page and returns it.

    This function uses the pandas library to read the first HTML table from the Wikipedia page
    'List of S&P 500 companies'

    Returns:
    - pd.DataFrame: A pandas DataFrame containing the S&P 500 companies table.
    """

    sp500_table = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
    return sp500_table

def filter_data(prices_df):
    """
    Filters out columns with one or more NA values from the input DataFrame.

    Parameters:
    - prices_df (pd.DataFrame): A pandas DataFrame with potential NA values in its columns.

    Returns:
    - pd.DataFrame: A DataFrame with columns containing NA values removed.
    """
    # Identify columns that do not have any NA values.
    # It checks for non-NA values across rows (for each column) using notna(),
    # and then ensures that every value within the column is True for non-NA (using all(axis=0)).
    columns_without_na = prices_df.columns[prices_df.notna().all(axis=0)]

    # Create a new DataFrame that contains only the columns without NA values.
    # This is done by using the columns identified in the previous step to filter the input DataFrame.
    df_filtered = prices_df[columns_without_na]

    # Return the DataFrame with columns without any NA values.
    return df_filtered

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

    # Reshape the data for sklearn. Independent variable needs to be in the format of
    # [[x1], [x2], [x3], ...] even if it's a single feature.
    X = series_independent.values.reshape(-1, 1)
    y = series_dependent.values

    # Perform linear regression using sklearn's LinearRegression class
    lm_model = LinearRegression()
    lm_model.fit(X, y)  # Fit the model to the data

    # Return the intercept and slope of the regression line
    # lm_model.intercept_ is a scalar, while lm_model.coef_ is an array.
    # Since we are performing simple linear regression, we return lm_model.coef_[0] to get the slope value.
    return lm_model.intercept_, lm_model.coef_[0]

def plot_regression_lines(mu, gamma, y1, y2,  pct_training= 2/3):
    """
    Plot the regression lines between two pandas Series and mark the end of the training period.

    Parameters:
    mu (float or pandas.Series): The intercept term from the regression model.
    gamma (float or pandas.Series): The slope term from the regression model.
    y1 (pandas.Series): Dependent variable data series.
    y2 (pandas.Series): Independent variable data series.
    training_period_ratio (float): The ratio of the training period to the total period (default is 2/3).
    """
    # Ensure that mu and gamma are broadcastable to the size of y1/y2 if they are scalar
    mu = pd.Series(mu, index=y1.index) if np.isscalar(mu) else mu
    gamma = pd.Series(gamma, index=y1.index) if np.isscalar(gamma) else gamma

    # Combine y1 and the linear combination of mu and gamma * y2 into a DataFrame
    tmp = pd.concat([y1, mu + gamma * y2], axis=1)
    tmp.columns = [y1.name,  f"mu + gamma x {y2.name}"]

    # Plot
    plt.figure(figsize=(10, 5))
    plt.plot(tmp.index, tmp.iloc[:, 0], label=tmp.columns[0])  # y1
    plt.plot(tmp.index, tmp.iloc[:, 1], label=tmp.columns[1])  # mu + gamma * y2

    # Determine the index for the end of the training period
    T_trn_index = int(len(y1) * pct_training)

    # Add a vertical line for the training period end
    plt.axvline(x=y1.index[T_trn_index], color='blue', lw=2, label='End of Training Period')

    # Set the x-axis locator and formatter if y1 has a datetime index
    if isinstance(y1.index, pd.DatetimeIndex):
        ax = plt.gca()
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    # Set tick positions if the index is not datetime
    else:
        plt.xticks(np.arange(0, len(y1), step=max(1, len(y1)//10)))

    plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
    plt.tight_layout()  # Adjust layout to prevent clipping of tick-labels

    # Add title, legend, and show plot
    plt.title('Regression of y1 from y2')
    plt.legend(loc='upper left')
    plt.show()