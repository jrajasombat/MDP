import pandas as pd
import numpy as np
from scipy.optimize import minimize, Bounds
import datetime
import dateutil.relativedelta
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st



def import_prices_returns(coi):
    '''This function simply reads a CSV file and fills up required dataframes.'''

    # Import prices. NOTE THE LOCATION OF THE CSV FILE
    prices_all = pd.read_csv('Prices7.csv', header=0, index_col=0, na_values=1, parse_dates=True)
    prices_all = prices_all.dropna()

    # Compute returns
    returns_all = prices_all.pct_change()
    returns_all = returns_all.dropna()

    # Retain columns of interest (coi)
    prices = prices_all[coi]
    returns = returns_all[coi]

    return prices, returns



def ask_for_input_variables():
    '''A function that will ask the user for input variables.'''

    interval = st.sidebar.selectbox('Enter rebalancing interval', ('Annually', 'Quarterly','Monthly','Weekly'))
    time_span = st.sidebar.slider('Enter the time span in years', 1, 20, 10, 1)
    covariance_look_back = st.sidebar.slider('Enter covariance look back in years', 1, 20, 5, 1)

    return interval, time_span, covariance_look_back




def get_book_end_dates(time_span, prices):
    '''This subfunction retrieves the last date in the returns dataframe and then
    find the start date that corresponds to the time span that is inputted.
    When the start date that corresponds to the time span is not in the returns
    dataframe, this subfunction will retrieve the very first date just prior the start date.'''

    end_date = prices.tail(1).index[0]
    start_date = end_date + dateutil.relativedelta.relativedelta(years=-time_span)
    start_date = prices.iloc[prices.index.get_loc(datetime.datetime(start_date.year,start_date.month,start_date.day), method='pad')].name
    return start_date, end_date



def get_reset_dates(interval, start_date, prices):
    '''This subfunction will retrieve the dates when rebalancing occurs.
    Rebalancing dates will be defined by the interval that is specified.
    At the start date, the portfolio will be equal weighted. After each interval,
    the portfolio will be rebalanced based on a covariance matrix and weights will
    be adjusted accordingly. Note that the data frequency is weekly.'''

    if interval == 'Weekly':
        time_step = 1
    elif interval == 'Monthly':
        time_step = 4
    elif interval == 'Quarterly':
        time_step = 12
    elif interval == 'Annually':
        time_step = 52
    else:
        print('Interval not valid')
    reset_dates = prices.truncate(before = start_date)
    reset_dates = reset_dates.iloc[::time_step, :]
    return reset_dates.index



def calc_diversification_ratio(w, V):
    '''Ben Helveys code'''

    # average weighted vol
    w_vol = np.dot(np.sqrt(np.diag(V)), w.T)
    # portfolio vol
    port_vol = (w.T @ V @ w)**0.5
    diversification_ratio = w_vol/port_vol
    # return negative for minimization problem (maximize = minimize -)
    return -diversification_ratio



def max_div_port(w0, V, bnd=None, long_only=False):
    '''Ben Helveys code'''

    # w0: initial weight
    # V: covariance matrix
    # bnd: individual position limit
    # long only: long only constraint
    opt_bounds = Bounds(0, 1)
    cons = {'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1}
    res = minimize(calc_diversification_ratio, w0, bounds=opt_bounds, args=V, method='SLSQP', options={'disp': False}, constraints=cons)
    return res['x']



def calc_portfolio_weights(covariance_look_back, reset_dates, returns, prices):
    '''This subfunction will output the portfolio weights at each reset date.
    Weights will be determined by the covariance matrices at reset dates which
    in turn are computed based on the look back period. Beginning weights are
    equal weights.'''

    #Create dataframe from empty dictionary
    temp_list = []
    for i in range(len(reset_dates)):
        temp_list.append(None)
    temp_dict = dict(zip(reset_dates, temp_list))
    portfolio_weights = pd.DataFrame([temp_dict]).transpose().rename(columns={0:'Weights'})

    #Get beginning weights (i.e., equal weights)
    number_of_asset_classes = returns.shape[1]
    beginning_weights = np.repeat(1 / number_of_asset_classes, number_of_asset_classes)
    portfolio_weights.iloc[0]['Weights'] = beginning_weights.round(5)

    #Loop over remaining reset dates
    #Construct covariance matrices
    #Calculate diversification ratio
    #Compute MDP weights
    for i in list(range(len(reset_dates)))[1:]:
        covariance_start_date = reset_dates[i] + dateutil.relativedelta.relativedelta(years=-covariance_look_back)
        covariance_start_date = prices.iloc[prices.index.get_loc(datetime.datetime(covariance_start_date.year,covariance_start_date.month,covariance_start_date.day), method='pad')].name
        covariance_matrix = returns[covariance_start_date:reset_dates[i]].cov()

        initial_weights = portfolio_weights.loc[reset_dates[i-1]]
        initial_weights_array = np.array(initial_weights.values.tolist())
        ending_weights = max_div_port(w0=initial_weights_array[0], V=covariance_matrix, bnd=None, long_only=True)

        portfolio_weights.iloc[i]['Weights'] = ending_weights.round(5)

    return portfolio_weights



def rebalance(coi):
    '''This function combines all the subfunctions and outputs
    the weights at each iterval and graphs the changes to the weights.'''

    #Call subfunctions
    prices, returns = import_prices_returns(coi)
    interval, time_span, covariance_look_back = ask_for_input_variables()
    start_date = get_book_end_dates(time_span, prices)[0]
    reset_dates = get_reset_dates(interval, start_date, prices)
    weights = calc_portfolio_weights(covariance_look_back, reset_dates, returns, prices)

    #Create new dataframe to house the computed weights
    temp_dict = {}

    for j in range(len(coi)):
        temp_list = []
        temp_dict[coi[j]] = temp_list
        for i in range(len(weights)):
            temp_list.append(weights.iloc[i][0][j])
        temp_dict[coi[j]] = temp_list

    temp_dict['Date'] = weights.index
    portfolio_weights = pd.DataFrame(temp_dict)
    portfolio_weights['Date'] = portfolio_weights['Date'].dt.strftime('%Y/%m/%d')
    portfolio_weights.set_index('Date', inplace = True)

    #Graph the changes in weights
    st.markdown('## Figure 1: Change in portfolio weights over time (%)')
    #sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize = (20,10))
    ax = sns.lineplot(data = portfolio_weights, linewidth = 3)
    ax.set_xticklabels(labels = portfolio_weights.index, rotation = 90, fontsize = 20)
    ax.set_yticklabels(labels = list(np.arange(0,110,10)), fontsize = 20)
    ax.set_xlabel('', fontsize = 18)
    ax.set_ylabel ('', fontsize = 18)
    ax.legend(loc='upper left', fontsize = 18);
    st.pyplot(fig)

    #Show the final weights in a dataframe
    st.markdown('## ')
    st.markdown('## Figure 2: Portfolio weights at each reset')
    st.dataframe(portfolio_weights, height = 1500, width = 500)

    return portfolio_weights
