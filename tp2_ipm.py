import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import skew, kurtosis
from statsmodels.regression.rolling import RollingOLS


def plot_graph(currency_str: str, currency: str, filepath: str, betas):
    plt.plot(betas.index, betas)
    plt.title(fr" Rolling UIP regression for the {currency_str} pair (5-year rolling windows)")
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
    # Identify currency
    curr = currency_list[i]
    currency_string = curr + "/USD"

    # Prepare regression inputs
    delta = log_exchange_rate_change[curr]
    delta = delta[~(delta.index > '2020-10-31')]
    y = delta.to_numpy()

    fp = forward_discount[curr]
    x = fp.to_numpy()[:-1].copy()
    x = sm.add_constant(x)

    # Initialize and run model
    model = sm.OLS(y, x)
    results = model.fit()

    # Extract outputs and coefficients
    constant = results.params[0]
    constant_t_value = results.tvalues[0]
    beta = results.params[1]
    beta_t_value = results.tvalues[1]
    r_squared = results.rsquared

    # Prepare results' dataframe to append
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

# Portfolio #3
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
y54 = df_betas["average_return"].astype(float) * 12

x54 = df_betas[["rx", "hml"]]

model54 = sm.OLS(y54.astype(float), x54.astype(float))
results54 = model54.fit()

print(results54.summary())


"""
Q6 - Stock returns and USD betas
"""

""" 6.1 - Regressions """
# Load data
df_names = pd.read_csv(filepath_or_buffer=rf"{file_path}\firm_950_names.csv", sep=",", )
df_returns = pd.read_csv(filepath_or_buffer=rf"{file_path}\firms_950_rets_prices.csv", sep=",", )

# Prepare data
df_returns['DateTime'] = pd.to_datetime(df_returns['date'].astype(str), format='%Y%m%d')
df_returns = df_returns.set_index("DateTime")
df_ret_q6 = df_returns[~(df_returns.index < '2000-01-01')]
df_ret_q6 = df_ret_q6[~(df_ret_q6.index > '2016-01-01')]
df_ret_q6 = df_ret_q6[["permno", "ret"]]

dollar_factor_q6 = dollar_factor_ret.copy().to_frame()
dollar_factor_q6 = dollar_factor_q6[~(dollar_factor_q6.index < '2000-01-01')]
dollar_factor_q6 = dollar_factor_q6[~(dollar_factor_q6.index > '2016-01-01')]
dollar_factor_q6.index = dollar_factor_q6.index.astype('datetime64[ns]')

firm_list = df_ret_q6.copy()
firm_list = firm_list["permno"].unique().tolist()

# Regression
betas_q6 = []
for i in range(len(firm_list)):
    firm = firm_list[i]
    firm_ret = df_ret_q6[df_ret_q6["permno"] == firm]
    y6 = firm_ret["ret"].to_numpy()

    x6 = dollar_factor_q6.to_numpy()
    x6 = sm.add_constant(x6)

    if np.isnan(y6).any():  # drop firm if returns contain NaN
        pass
    else:

        model6 = sm.OLS(y6, x6)
        results6 = model6.fit()

        ret_beta = results6.params[1]

        results_columns_q6 = ["permno", "ret_beta", ]
        results_array_q6 = np.array([firm, ret_beta, ])
        results_q6 = pd.DataFrame(results_array_q6.reshape(-1, len(results_array_q6)), columns=results_columns_q6, )

        betas_q6.append(results_q6)

df_betas_q6 = pd.concat(betas_q6, axis=0).reset_index(drop=True, )

""" 6.2 & 6.3 - Ten portfolios """
# Rank firms by betas
df_ranked = df_betas_q6.copy()
df_ranked["rank"] = df_ranked["ret_beta"].rank(ascending=True, method='first', )
df_ranked['group'] = pd.cut(df_ranked['rank'], 10, labels=False) + 1

# Create unique date list and prepare data for return calculation
df_ret_oos = df_returns[~(df_returns.index < '2015-02-01')]
df_ret_oos = df_ret_oos[~(df_ret_oos.index > '2020-11-01')]

# Create list for iteration purpose
date_list = df_ret_oos.index.unique().to_list()
group_list = np.arange(1, 11, 1).tolist()

returns_by_pf = []
# Loop to calculate returns
for i in range(len(group_list)):
    # Identify Portfolio
    group = group_list[i]
    group_name = fr"Portfolio {group}"

    # Identify firms in identified portfolio
    group_firm = df_ranked.copy()
    group_firm_list = group_firm["permno"][group_firm["group"] == group].to_list()

    # Filter df_ret_oos and firm
    df_ret_mc = df_ret_oos[df_ret_oos["permno"].isin(group_firm_list)]

    for j in range(len(date_list)):
        # Identify date
        dt = date_list[j]

        # Filter df_ret_oos by date and firm
        df_ret_mc_dt = df_ret_mc[df_ret_mc.index == dt]

        # Calculate Market Cap
        df_weight = df_ret_mc_dt.copy()
        df_weight["market_cap"] = df_weight["prc"] * df_weight["shrout"]

        # Calculate weight by market capitalization
        df_weight["weight"] = df_weight["market_cap"] / df_weight["market_cap"].sum()

        # Calculate weighted returns
        df_weight["weighted_returns"] = df_weight["weight"] * df_weight["ret"]
        total_return_by_date_portfolio = df_weight["weighted_returns"].sum()

        # Merge Beta
        df_beta_merged = df_weight.merge(df_betas_q6, how='left', left_on='permno', right_on='permno', indicator=False, )
        df_beta_merged["weighted_beta"] = df_beta_merged["weight"] * df_beta_merged["ret_beta"]
        total_weight_beta_by_date_portfolio = df_beta_merged["weighted_beta"].sum()

        # Create frame to append
        results_columns_q63 = ["Portfolio", "Date", "Return", "Beta", ]
        results_array_q63 = np.array([group_name, dt, total_return_by_date_portfolio, total_weight_beta_by_date_portfolio])
        results_q63 = pd.DataFrame(results_array_q63.reshape(-1, len(results_array_q63)), columns=results_columns_q63, )

        returns_by_pf.append(results_q63)


returns_by_pf_date = pd.concat(returns_by_pf, axis=0).set_index("Date")


""" 6.4 - Correlation """
# Prepare dollar factor date
dollar_factor_q64 = dollar_factor_ret.copy().to_frame()
dollar_factor_q64 = dollar_factor_q64[~(dollar_factor_q64.index < '2015-02-01')]
dollar_factor_q64 = dollar_factor_q64[~(dollar_factor_q64.index > '2020-11-01')]
dollar_factor_q64.index = dollar_factor_q64.index.astype('datetime64[ns]')


correlation_by_pf = []
# Loop to calculate returns
for i in range(len(group_list)):
    # Identify Portfolio
    group = group_list[i]
    group_name = fr"Portfolio {group}"

    # Filter for portfolio
    df_ret_by_pf = returns_by_pf_date[returns_by_pf_date["Portfolio"] == group_name]

    # Calculate correlation
    corr = np.corrcoef(df_ret_by_pf["Return"].astype("float"), dollar_factor_q64[0])
    corr_coef = corr[0, 1, ]

    # Create frame to append
    results_columns_q64 = ["Portfolio", "Correlation", ]
    results_array_q64 = np.array([group_name, corr_coef, ])
    results_q64 = pd.DataFrame(results_array_q64.reshape(-1, len(results_array_q64)), columns=results_columns_q64, )

    correlation_by_pf.append(results_q64)

df_correlation = pd.concat(correlation_by_pf, axis=0)
