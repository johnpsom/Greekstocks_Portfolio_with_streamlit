import csv
import requests as req
from contextlib import closing
import pandas as pd
import numpy as np
from scipy import stats
import pypfopt
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import expected_returns,risk_models #,plotting
from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices
from pypfopt.risk_models import CovarianceShrinkage

import io
import base64
import os
import json
import pickle
import uuid
import re
from enum import Enum
from io import BytesIO, StringIO
from typing import Union

# Momentum score function
def momentum_score(ts):
    x = np.arange(len(ts))
    log_ts = np.log(ts)
    regress = stats.linregress(x, log_ts)
    annualized_slope = (np.power(np.exp(regress[0]), 252) -1) * 100
    return annualized_slope * (regress[2] ** 2)

def returns_from_prices(prices, log_returns=False):
    """
    Calculate the returns given prices.
    :param prices: adjusted (daily) closing prices of the asset, each row is a
                   date and each column is a ticker/id.
    :type prices: pd.DataFrame
    :param log_returns: whether to compute using log returns
    :type log_returns: bool, defaults to False
    :return: (daily) returns
    :rtype: pd.DataFrame
    """
    if log_returns:
        ret= np.log(1 + prices.pct_change())
    else:
        ret= prices.pct_change()
    return ret

  
def capm_returns(prices, market_prices=None, returns_data=False, risk_free_rate=0.02, \
                 compounding=True, frequency=252):
    """
    Compute a return estimate using the Capital Asset Pricing Model. Under the CAPM,
    asset returns are equal to market returns plus a :math:`\beta` term encoding
    the relative risk of the asset.
    .. math::
        R_i = R_f + \\beta_i (E(R_m) - R_f)
    :param prices: adjusted closing prices of the asset, each row is a date
                    and each column is a ticker/id.
    :type prices: pd.DataFrame
    :param market_prices: adjusted closing prices of the benchmark, defaults to None
    :type market_prices: pd.DataFrame, optional
    :param returns_data: if true, the first arguments are returns instead of prices.
    :type returns_data: bool, defaults to False.
    :param risk_free_rate: risk-free rate of borrowing/lending, defaults to 0.02.
                           You should use the appropriate time period, corresponding
                           to the frequency parameter.
    :type risk_free_rate: float, optional
    :param compounding: computes geometric mean returns if True,
                        arithmetic otherwise, optional.
    :type compounding: bool, defaults to True
    :param frequency: number of time periods in a year, defaults to 252 (the number
                        of trading days in a year)
    :type frequency: int, optional
    :return: annualised return estimate
    :rtype: pd.Series
    """
    if not isinstance(prices, pd.DataFrame):
        warnings.warn("prices are not in a dataframe", RuntimeWarning)
        prices = pd.DataFrame(prices)
    if returns_data:
        returns = prices
        market_returns = market_prices
    else:
        returns = returns_from_prices(prices)
        if market_prices is not None:
            market_returns = returns_from_prices(market_prices)
        else:
            market_returns = None
    # Use the equally-weighted dataset as a proxy for the market
    if market_returns is None:
        # Append market return to right and compute sample covariance matrix
        returns["mkt"] = returns.mean(axis=1)
    else:
        market_returns.columns = ["mkt"]
        returns = returns.join(market_returns, how="left")
    # Compute covariance matrix for the new dataframe (including markets)
    cov = returns.cov()
    # The far-right column of the cov matrix is covariances to market
    betas = cov["mkt"] / cov.loc["mkt", "mkt"]
    betas = betas.drop("mkt")
    # Find mean market return on a given time period
    if compounding:
        mkt_mean_ret = (1 + returns["mkt"]).prod() ** (
            frequency / returns["mkt"].count()
        ) - 1
    else:
        mkt_mean_ret = returns["mkt"].mean() * frequency
    # CAPM formula
    return risk_free_rate + betas * (mkt_mean_ret - risk_free_rate)

#select stocks columns
def select_columns(data_frame, column_names):
    new_frame = data_frame.loc[:, column_names]
    return new_frame

#cumulative returns calculation
def cumulative_returns(stock,returns):
    res = (returns + 1.0).cumprod()
    res.columns = [stock]
    return res


def download_button(object_to_download, download_filename, button_text, pickle_it=False):
    """
    Generates a link to download the given object_to_download.
    Params:
    ------
    object_to_download:  The object to be downloaded.
    download_filename (str): filename and extension of file. e.g. mydata.csv,
    some_txt_output.txt download_link_text (str): Text to display for download
    link.
    button_text (str): Text to display on download button (e.g. 'click here to download file')
    pickle_it (bool): If True, pickle file.
    Returns:
    -------
    (str): the anchor tag to download object_to_download
    Examples:
    --------
    download_link(your_df, 'YOUR_DF.csv', 'Click to download data!')
    download_link(your_str, 'YOUR_STRING.txt', 'Click to download text!')
    """
    if pickle_it:
        try:
            object_to_download = pickle.dumps(object_to_download)
        except pickle.PicklingError as e:
            st.write(e)
            return None

    else:
        if isinstance(object_to_download, bytes):
            pass

        elif isinstance(object_to_download, pd.DataFrame):
            object_to_download = object_to_download.to_csv(index=False)
            towrite = io.BytesIO()
            #object_to_download = object_to_download.to_excel(towrite, encoding='utf-8', index=False, header=True)
            towrite.seek(0)

        # Try JSON encode for everything else
        else:
            object_to_download = json.dumps(object_to_download)

    try:
        # some strings <-> bytes conversions necessary here
        b64 = base64.b64encode(object_to_download.encode()).decode()

    except AttributeError as e:
        b64 = base64.b64encode(towrite.read()).decode()

    button_uuid = str(uuid.uuid4()).replace('-', '')
    button_id = re.sub('\d+', '', button_uuid)
    custom_css = f""" 
        <style>
            #{button_id} {{
                display: inline-flex;
                align-items: center;
                justify-content: center;
                background-color: rgb(255, 255, 255);
                color: rgb(38, 39, 48);
                padding: .25rem .75rem;
                position: relative;
                text-decoration: none;
                border-radius: 4px;
                border-width: 1px;
                border-style: solid;
                border-color: rgb(230, 234, 241);
                border-image: initial;
            }} 
            #{button_id}:hover {{
                border-color: rgb(246, 51, 102);
                color: rgb(246, 51, 102);
            }}
            #{button_id}:active {{
                box-shadow: none;
                background-color: rgb(246, 51, 102);
                color: white;
                }}
        </style> """

    dl_link = custom_css + f'<a download="{download_filename}" id="{button_id}" href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}">{button_text}</a><br></br>'
    return dl_link


def backtest_portfolio(df,dataset=1000,l_days=700,momentum_window=120,minimum_momentum=70,portfolio_size=5,tr_period=5,cutoff=0.05,port_value=10000,a_v=0):
    allocation={}
    non_trading_cash=0
    new_port_value=0
    print(momentum_window,minimum_momentum,portfolio_size,tr_period,cutoff,port_value)
    added_value=tr_period*a_v
    no_tr=1 #number of trades performed
    init_portvalue=port_value
    plotted_portval=[]
    plotted_ret=[]
    pval=pd.DataFrame(columns=['Date','portvalue','porteff'])
    keep_df_buy=True
    for days in range(dataset,len(df),tr_period):
        df_tr=df.iloc[days-l_days:days,:]
        df_date=datetime.strftime(df.iloc[days,:].name,'%d-%m-%Y')
        if days<=dataset:
            ini_date=df_date
        if days>dataset and keep_df_buy is False:
            latest_prices = get_latest_prices(df_tr)
            new_port_value=non_trading_cash
            allocation=df_buy['shares'][:-1].to_dict()
            #print(allocation)
            if keep_df_buy is False:
                #print('Sell date',df_date)
                for s in allocation:
                    new_port_value=new_port_value+allocation.get(s)*latest_prices.get(s)
                    #print('Sell ',s,'stocks: ',allocation.get(s),' bought for ',df_buy['price'][s],' sold for ',latest_prices.get(s)
                    #       ,' for total:{0:.2f} and a gain of :{1:.2f}'.format(allocation.get(s)*latest_prices.get(s),
                    #      (latest_prices.get(s)-df_buy['price'][s])*allocation.get(s)))
                eff=new_port_value/port_value-1
                #print('Return after trading period {0:.2f}%  for a total Value {1:.2f}'.format(eff*100,new_port_value))
                port_value=new_port_value
                plotted_portval.append(round(port_value,2))
                plotted_ret.append(round(eff*100,2))
                pval=pval.append({'Date':df_date,'portvalue':round(port_value,2),'porteff':round(eff*100,2)}, ignore_index=True)
                port_value=port_value+added_value #add 200 after each trading period
        df_m=pd.DataFrame()
        m_s=[]
        st=[]
        for s in tickers_gr:
            st.append(s)
            m_s.append(momentum_score(df_tr[s].tail(momentum_window)))
        df_m['stock']=st
        df_m['momentum']=m_s
        dev=df_m['momentum'].std()
        # Get the top momentum stocks for the period
        df_m = df_m.sort_values(by='momentum', ascending=False)
        df_m=df_m[(df_m['momentum']>minimum_momentum-0.5*dev)&(df_m['momentum']<minimum_momentum+1.9*dev)].head(portfolio_size)
        # Set the universe to the top momentum stocks for the period
        universe = df_m['stock'].tolist()
        #print('universe',universe)
        # Create a df with just the stocks from the universe
        if len(universe)>2 and port_value>0 :
            keep_df_buy=False
            df_buy= get_portfolio(universe, df_tr, port_value, cutoff, df_m)
            #print('Buy date',df_date)
            #print(df_buy)
            #print('trade no:',no_tr,' non allocated cash:{0:.2f}'.format(non_trading_cash),'total invested:', df_buy['value'].sum())
            port_value=df_buy['value'].sum()
            no_tr=no_tr+1
            #st_day=st_day+tr_period
        else:
            #print('Buy date',df_date,'Not enough stocks in universe to create portfolio',port_value)
            port_value=port_value+added_value
            keep_df_buy=True

    total_ret=100*(new_port_value/(init_portvalue+no_tr*added_value)-1)
    dura_tion=(no_tr-1)*tr_period
    if no_tr>2:
        #print('Total return: {0:.2f} in {1} days'.format(total_ret,dura_tion))
        #print('Cumulative portfolio return:',round(list(pval['porteff'].cumsum())[-1],2))
        #print('total capital:',init_portvalue+no_tr*added_value, new_port_value)
        tot_contr=init_portvalue+no_tr*added_value
        s=round(pd.DataFrame(plotted_portval).pct_change().add(1).cumprod()*10,2)
        rs={'trades':no_tr,'momentum_window':momentum_window,
                        'minimum_momentum':minimum_momentum,
                        'portfolio_size':portfolio_size,
                        'tr_period':tr_period,
                        'cutoff':cutoff,
                        'tot_contribution':tot_contr,'final port_value':new_port_value,
                        'cumprod':s[-1:][0].values[0], 'tot_ret':total_ret,'drawdown':s.diff().min()[0]}
        
        return rs

def rebalance_portfolio(df_old,df_new):
    '''rebalance old with new proposed portfolio'''
    old_port_value=df_old['value'].sum()
    new_port_value=old_port_value
    new_stocks= list(df_old.stock[:-1]) + list(set(df_new.stock[:-1])-set(df_old.stock))
    for stock in new_stocks:
        #close old positions that do not appear in new portfolio
        if (stock in list(df_old.stock)) and (stock not in list(df_new.stock[:-1])) :
            #close positions
            if df_old.loc[df_old.stock==stock,'shares'].values[0]>0:
                st.write(f'κλείσιμο θέσης στην μετοχή {stock}')
                new_port_value=new_port_value+df_old.loc[df_old.stock==stock,'shares'].values[0]
            if df_old.loc[df_old.stock==stock,'shares'].values[0]<0:
                st.write(f'κλείσιμο θέσης στην μετοχή {stock}')
                new_port_value=new_port_value+df_old.loc[df_old.stock==stock,'shares'].values[0]
        #open new positions that only appear in new portfolio
        if stock in list(set(df_new.stock[:-1])-set(df_old.loc[:,'stock'])):
            if df_new.loc[df_new.stock==stock,'shares'].values[0]>0:
                st.write(f'Αγόρασε {df_new.loc[df_new.stock==stock,"shares"].values[0]} μετοχές της {stock} για να ανοιχτεί νέα long θέση')
            if df_new.loc[df_new.stock==stock,'shares'].values[0]<0:
                st.write(f'Πούλησε {df_new.loc[df_new.stock==stock,"shares"].values[0]} μετοχές της {stock} για να ανοιχτεί νέα short θέση')
        #modify positions of stocks that appear in new and old portfolio
        if (stock in list(df_old.stock)) and (stock in list(df_new.stock[:-1])):
            #change positions
            if df_new.loc[df_new.stock==stock,'shares'].values[0]>0 and df_old.loc[df_old.stock==stock,'shares'].values[0]>0:
                new_shares=df_new.loc[df_new.stock==stock,"shares"].values[0]-df_old.loc[df_old.stock==stock,'shares'].values[0]
                if new_shares>=0:
                    st.write(f'Αγόρασε ακόμη {round(new_shares,0)} της μετοχής {stock}')
                if new_shares<0:
                    st.write(f'Πούλησε ακόμη {round(-new_shares,0)} της μετοχής {stock}')
            if df_new.loc[df_new.stock==stock,'shares'].values[0]<0 and df_old.loc[df_old.stock==stock,'shares'].values[0]<0:
                new_shares=df_new.loc[df_new.stock==stock,"shares"].values[0]-df_old.loc[df_old.stock==stock,'shares'].values[0]
                if new_shares>=0:
                    st.write(f'Αγόρασε ακόμη {round(new_shares,0)} της μετοχής {stock}')
                if new_shares<0:
                    st.write(f'Πούλησε ακόμη {round(-new_shares,0)} της μετοχής {stock}')
            if df_new.loc[df_new.stock==stock,'shares'].values[0]*df_old.loc[df_old.stock==stock,'shares'].values[0] < 0:
                new_shares=df_new.loc[df_new.stock==stock,'shares'].values[0] - df_old.loc[df_old.stock==stock,'shares'].values[0]
                if new_shares>=0:
                    st.write(f'Αγόρασε ακόμη {round(new_shares,0)} της μετοχής {stock}')
                if new_shares<0:
                    st.write(f'Πούλησε ακόμη {round(-new_shares,0)} της μετοχής {stock}')
    return new_port_value 
