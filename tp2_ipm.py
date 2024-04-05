import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.regression.rolling import RollingOLS
import matplotlib.pyplot as plt


def plot_graph(currency_str: str, currency: str, filepath: str, betas):
    plt.plot(betas.index, betas)
    plt.title(fr" Rolling UIP regression for the {currency_str} pair (5-year rolling windows")
    plt.xlabel("Date")
    plt.ylabel("Coefficient of the forward premium")
    # plt.show()
    plt.savefig(fr'{filepath}\outputs\q2_{currency}.png')
    plt.clf()


file_path = r"C:\Users\Client\OneDrive - HEC Montréal\International Porfolio Management\TP2"
currency_list = ["CAD", "JPY", "SEK", "CHF", "EUR", "GBP", ]

# todo: add comment for each line and/or subsection of the code


"""
Import spots and forward data
"""
df_spot = pd.read_csv(filepath_or_buffer=rf"{file_path}\spot.csv", sep=";", )
df_forward = pd.read_csv(filepath_or_buffer=rf"{file_path}\forward.csv", sep=";", )

"""
 Preliminaries
"""
# Order rates by date
df_spot = df_spot.sort_values("Date").set_index("Date").dropna()
df_forward = df_forward.sort_values("Date").set_index("Date").dropna()

# Create log series
spot_log = np.log(df_spot)
forward_log = np.log(df_forward)

# Calculate log exchange rate change
spot_log_shifted = spot_log.shift(periods=1)
log_exchange_rate_change = spot_log_shifted.subtract(spot_log, axis=1).dropna()

# Calculate log excess returns and log spot returns
forward_log_shifted = forward_log.shift(periods=1)
log_excess_returns = forward_log_shifted.subtract(spot_log, axis=1).dropna()

# Drop rows after 2020-10
log_exchange_rate_change = log_exchange_rate_change[~(log_exchange_rate_change.index > '2020-10-31')]
log_excess_returns = log_excess_returns[~(log_excess_returns.index > '2020-10-31')]

"""
Question 1 - Full-sample UIP regressions
"""
# Calculate forward premium
forward_premium = forward_log.subtract(spot_log, axis=1, )
forward_premium = forward_premium[~(forward_premium.index > '2020-10-31')]

# Regression by currency
df_results_appended_q1 = []
for i in range(len(currency_list)):
    curr = currency_list[i]
    currency_string = curr + "/USD"

    delta = log_exchange_rate_change[curr]
    y = delta.to_numpy()

    fp = forward_premium[curr]
    x = fp.to_numpy()[:-1].copy()
    x = sm.add_constant(x)

    model = sm.OLS(y, x)
    results = model.fit()

    constant = results.params[0]
    constant_t_value = results.tvalues[0]
    beta = results.params[1]
    beta_t_value = results.tvalues[1]
    r_squared = results.rsquared

    results_columns_q1 = ["Currency", "Constant", "Constant t-value", "Beta", "Beta t-value", "R-squared"]
    results_array_q1 = np.array([currency_string, constant, constant_t_value, beta, beta_t_value, r_squared])
    results_q1 = pd.DataFrame(results_array_q1.reshape(-1, len(results_array_q1)), columns=results_columns_q1, )

    df_results_appended_q1.append(results_q1)

q1_results = pd.concat(df_results_appended_q1, axis=0)


"""
Question 2 - Rolling UIP regressions
"""
# Regression by currency
for i in range(len(currency_list)):
    curr = currency_list[i]
    currency_string = curr + "/USD"

    delta = log_exchange_rate_change[curr]
    y = delta

    fp = forward_premium[curr]
    x_var = fp[1:].copy()
    x = sm.add_constant(x_var)

    model_q2 = RollingOLS(y, x, window=60, )
    results_q2 = model_q2.fit()

    varr = results_q2.params[curr].dropna()

    varr.index = pd.to_datetime(varr.index)

    plot_graph(currency_str=currency_string,
               currency=curr,
               filepath=file_path,
               betas=varr, )
