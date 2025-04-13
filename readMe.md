# Run

While being in the root directory, run this to have all imports

```cmd
python -m src.trial
```

# Data

Explain each stages dict structure

# Feature engineering

These functions can be combined to reduce the number of passes. However, doing so will decrease code readability. If performance is a priority, consider modifying the functions.

## Paper

```python
# Momentum Variables

def calculate_momentum(months, prices_monthly, offset_start, offset_end):

def calculate_mom1m(months, prices_monthly):

def calculate_mom12m(months, prices_monthly):

def calculate_mom36m(months, prices_monthly):

def calculate_chmom(months, prices_monthly):

def calculate_maxret(months, max_daily_returns_monthly):

def calculate_indmom(stocks, sic_codes):

def handle_indmom(stock, indmom):

# Liquidity Variables

def calculate_turn(months, vol_monthly, shares_monthly):

def calculate_std_turn(prices_daily, shares):

def calculate_mve(months, market_cap_monthly):

def calculate_dolvol(months, dollar_volume_monthly):

def calculate_ill(prices_daily):

def calculate_zerotrade(months, vol_sum, shares_monthly, zero_trading_days, trading_days_count):

# Risk Measures:

def calculate_retvol(prices_daily):

def calculate_idiovol(months, month_latest_week, weekly_returns, market_weekly_returns, interval=156, increment=4):

def calculate_beta_betasq(months, month_latest_week, weekly_returns, market_weekly_returns, interval=156, increment=4):

# Valuation Ratios and Fundamental Signals:

def calculate_ep_sp(income_statement_annual, market_caps):

def calculate_agr(balance_sheet_annual):
```

## Variants

```python
# EOD and Market Cap ensured to be sorted

# Difference is rather than months 1-12 we get 0-11
def calculate_mom12m_current(months, prices_monthly):

# Difference is t_1 = months[i] rather than t_1 = months[i+1]
def calculate_chmom_current(months, prices_monthly):

# Difference is we get the max_return from current month. Same as the function "get_max_daily_returns_monthly()"
def calculate_maxret_current(prices_daily):

# Liquidity Variables

# Difference is we get the market cap of the last trading day from current month.
# Same as the function "get_market_cap_monthly()" but using natural log
def calculate_mve_current(market_caps):

# Difference is we get the dolvol of current month
# avg_dv = dollar_volume_monthly[curr_month]["sum"] / dollar_volume_monthly[curr_month]["count"]
def calculate_dolvol_current(months, dollar_volume_monthly):

# Difference is we get zerotrade of current month
# zero_days = zero_trading_days[current_month]
def calculate_zerotrade_current(months, vol_sum, shares_monthly, zero_trading_days, trading_days_count):

# Risk Measures:

# Differnce is the window starts from current month
# month_start = months[current]
def get_rolling_weekly_returns_current(months, month_latest_week, weekly_returns, interval=156, increment=4):


# Difference is the window starts from current month
# month_start = months[month_current_index]
def calculate_idiovol_current(months, month_latest_week, weekly_returns, market_weekly_returns, interval=156, increment=4):

# Difference is beta and betasq from this month
# month_start = months[current]
def calculate_beta_betasq_current(months, month_latest_week, weekly_returns, market_weekly_returns, interval=156, increment=4):

def calculate_ep_sp_quarterly(income_statement_quarterly, market_caps):

def calculate_agr_quarterly(balance_sheet_quarterly):
```

# Notes

Financial Statements can in fact go earlier than the earliest eod, market cap dates since they were not public then.
