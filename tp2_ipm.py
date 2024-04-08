import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.regression.rolling import RollingOLS
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis


def plot_graph(currency_str: str, currency: str, filepath: str, betas):
    plt.plot(betas.index, betas)
    plt.title(fr" Rolling UIP regression for the {currency_str} pair (5-year rolling windows")
    plt.xlabel("Date")
    plt.ylabel("Coefficient of the forward premium")
    plt.savefig(fr'{filepath}\outputs\q2_{currency}.png')
    plt.clf()


def get_desc_stats(pf, name: str):
    mean = np.mean(pf)
    std_dev = np.std(pf)
    skewness = skew(pf)
    kurto = kurtosis(pf)
    sharpe_ratio = mean / std_dev

    columns = ["Portfolio", "Mean", "Standard Deviation", "Skewness", "Kurtosis", "Sharpe Ratio"]
    rez = np.array([name, mean, std_dev, skewness, kurto, sharpe_ratio])
    df_results = pd.DataFrame(rez.reshape(-1, len(rez)), columns=columns, )

    return df_results


file_path = r"C:\Users\Client\OneDrive - HEC Montréal\International Porfolio Management\TP2"
currency_list = ["CAD", "JPY", "SEK", "CHF", "EUR", "GBP", ]


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

"""
Question 1 - Full-sample UIP regressions
"""
# Calculate forward premium
forward_discount = forward_log.subtract(spot_log, axis=1, )
forward_discount = forward_discount[~(forward_discount.index > '2020-10-31')]

# Regression by currency
df_results_appended_q1 = []
for i in range(len(currency_list)):
    curr = currency_list[i]
    currency_string = curr + "/USD"

    delta = log_exchange_rate_change[curr]
    delta = delta[~(delta.index > '2020-10-31')]
    y = delta.to_numpy()

    fp = forward_discount[curr]
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
    delta = delta[~(delta.index > '2020-10-31')]
    y = delta

    fp = forward_discount[curr]
    x_var = fp.copy()
    x_var = x_var.shift(periods=1)
    x_var = x_var.dropna()
    x = sm.add_constant(x_var)

    model_q2 = RollingOLS(y, x, window=60, )
    results_q2 = model_q2.fit()

    beta_forward_premium = results_q2.params[curr].dropna()

    beta_forward_premium.index = pd.to_datetime(beta_forward_premium.index)

    plot_graph(currency_str=currency_string,
               currency=curr,
               filepath=file_path,
               betas=beta_forward_premium, )


"""
Question 3 - Carry trade portfolios
"""
# Allocate currencies to portfolio based on their forward discount
forward_discount_ranked = forward_discount.rank(ascending=False, method='first', axis=1)

# Portfolio #1
pf_one = forward_discount_ranked.copy()
pf_one[pf_one < 5] = 0
pf_one[pf_one > 0] = 0.5
pf_one_forward_discount = forward_discount.mul(pf_one, axis=1).sum(axis=1)

# Portfolio #2
pf_two = forward_discount_ranked.copy()
pf_two[pf_two > 4] = 0
pf_two[pf_two < 3] = 0
pf_two[pf_two > 0] = 0.5
pf_two_forward_discount = forward_discount.mul(pf_two, axis=1).sum(axis=1)

# Portfolio #2
pf_three = forward_discount_ranked.copy()
pf_three[pf_three > 2] = 0
pf_three[pf_three > 0] = 0.5
pf_three_forward_discount = forward_discount.mul(pf_three, axis=1).sum(axis=1)

# High Minus Low
hml_forward_discount = pf_three_forward_discount - pf_one_forward_discount

# Dollar Factor
dollar_factor_pf = forward_discount_ranked.copy()
dollar_factor_pf[dollar_factor_pf > 0] = 1 / 6
dollar_factor_forward_discount = forward_discount.mul(dollar_factor_pf, axis=1).sum(axis=1)


# Descriptive Statistics
one = get_desc_stats(pf=pf_one_forward_discount, name="Portfolio #1")
two = get_desc_stats(pf=pf_two_forward_discount, name="Portfolio #2")
three = get_desc_stats(pf=pf_three_forward_discount, name="Portfolio #3")
hml = get_desc_stats(pf=hml_forward_discount, name="High-minus-Low")
dollar_factor = get_desc_stats(pf=dollar_factor_forward_discount, name="Dollar Factor")

q3_results = pd.concat([one, two, three, hml, dollar_factor], axis=0)


"""
Question 4 - Momentum portfolios
"""
returns = log_exchange_rate_change.copy()
returns = returns[~(returns.index > '2020-10-31')]

# Allocate currencies to portfolio based on their lagged returns
lagged_returns_ranked = returns.rank(ascending=False, method='first', axis=1)
lagged_returns_ranked = lagged_returns_ranked.shift(periods=1)  # shift to match lagged return with current period

# Modify dataframes
lagged_returns_ranked = lagged_returns_ranked.dropna()
returns = returns[1:]

# Portfolio #1
pf_one = lagged_returns_ranked.copy()
pf_one[pf_one < 5] = 0
pf_one[pf_one > 0] = 0.5
pf_one_ret = returns.mul(pf_one, axis=1).sum(axis=1)

# Portfolio #2
pf_two = lagged_returns_ranked.copy()
pf_two[pf_two > 4] = 0
pf_two[pf_two < 3] = 0
pf_two[pf_two > 0] = 0.5
pf_two_ret = returns.mul(pf_two, axis=1).sum(axis=1)

# Portfolio #2
pf_three = lagged_returns_ranked.copy()
pf_three[pf_three > 2] = 0
pf_three[pf_three > 0] = 0.5
pf_three_ret = returns.mul(pf_three, axis=1).sum(axis=1)

# High Minus Low
hml_ret = pf_three_ret - pf_one_ret

# Dollar Factor
dollar_factor_pf = lagged_returns_ranked.copy()
dollar_factor_pf[dollar_factor_pf > 0] = 1 / 6
dollar_factor_ret = returns.mul(dollar_factor_pf, axis=1).sum(axis=1)


# Descriptive Statistics
one = get_desc_stats(pf=pf_one_ret, name="Portfolio #1")
two = get_desc_stats(pf=pf_two_ret, name="Portfolio #2")
three = get_desc_stats(pf=pf_three_ret, name="Portfolio #3")
hml = get_desc_stats(pf=hml_ret, name="High-minus-Low")
dollar_factor = get_desc_stats(pf=dollar_factor_ret, name="Dollar Factor")

q4_results = pd.concat([one, two, three, hml, dollar_factor], axis=0)


"""
Question 5 - Two-stage regressions
"""
# Load data
df_q5 = pd.read_csv(filepath_or_buffer=rf"{file_path}\q5_data.csv", sep=";", )

# Extract column name for iterations
columns_list = list(df_q5.columns)
portfolio_list = [x for x in columns_list if x.startswith("Portfolio")]

# Restrict Sample
df_q5 = df_q5.set_index("Dates")
df_q5.index = pd.to_datetime(df_q5.index)
df_q5 = df_q5[~(df_q5.index < '1984-12-01')]
df_q5 = df_q5[~(df_q5.index > '2020-10-31')]


""" 5.1 & 5.2 & 5.3"""
betas_q5 = []
for i in range(len(portfolio_list)):
    pf5 = portfolio_list[i]

    y5 = df_q5[pf5]

    x5 = df_q5[["rx", "hml"]]
    x5 = sm.add_constant(x5)

    model5 = sm.OLS(y5, x5)
    results5 = model5.fit()

    rx = results5.params["rx"]
    hml = results5.params["hml"]

    average_return = np.mean(y5)

    results_columns_q5 = ["Portfolio", "rx", "hml", "average_return"]
    results_array_q5 = np.array([pf5, rx, hml, average_return, ])
    results_q5 = pd.DataFrame(results_array_q5.reshape(-1, len(results_array_q5)), columns=results_columns_q5, )

    betas_q5.append(results_q5)

df_betas = pd.concat(betas_q5, axis=0).reset_index(drop=True, )

""" 5.4 """
y54 = df_betas["average_return"]

x54 = df_betas[["rx", "hml"]]
x54 = sm.add_constant(x54)

model54 = sm.OLS(y54.astype(float), x54.astype(float))
results54 = model54.fit()

print(results54.summary())


