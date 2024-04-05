import numpy as np
import pandas as pd
import statsmodels.api as sm

file_path = r"C:\Users\Client\OneDrive - HEC Montréal\International Porfolio Management\TP2"

currency_list = ["CAD", "JPY", "SEK", "CHF", "EUR", "GBP",]

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
results_q1 = []
for i in range(len(currency_list)):
    curr = currency_list[i]

    delta = log_exchange_rate_change[curr]
    Y = delta.to_numpy()

    fp = forward_premium[curr]
    X = fp.to_numpy()[:-1].copy()
    X = sm.add_constant(X)

    model = sm.OLS(Y, X)
    results = model.fit()

    constant = results.params[0]
    constant_t_value = results.tvalues[0]
    beta = results.params[1]
    beta_t_value = results.tvalues[1]
    r_squared = results.rsquared

    # todo: create dataframe from results (curr, constant, constant t value, beta, beta t value, r_squared)
    #  add prefix to currency before ("/USD")
    #  append results to results_q1