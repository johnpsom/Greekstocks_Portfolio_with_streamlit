# -*- coding: utf-8 -*-
"""
Created on Sun Oct 25 13:19:50 2020
@author: getyour.portfolio@gmail.com
"""
import streamlit as st
import csv
import requests as req
from contextlib import closing
from datetime import datetime
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

#import variables and functions used
from stocks import stocks
from greekstocks import momentum_score,returns_from_prices,capm_returns,get_latest_prices
from greekstocks import select_columns, cumulative_returns, download_button, rebalance_portfolio, backtest_portfolio

STYLE = """
<style>
img {
    max-width: 100%;
}
</style>
"""

FILE_TYPES = ["csv"]

class FileType(Enum):
    """Used to distinguish between file types"""
    CSV = "csv"

def get_file_type(file: Union[BytesIO, StringIO]) -> FileType:
    """The file uploader widget does not provide information on the type of file uploaded so we have
    to guess using rules or ML

    I've implemented rules for now :-)

    Arguments:
        file {Union[BytesIO, StringIO]} -- The file uploaded

    Returns:
        FileType -- A best guess of the file type
    """
    return FileType.CSV


@st.cache(ttl=24*60*60)
def load_data(tickers_gr):
    greekstocks_data={}
    close_data=pd.DataFrame()
    for ticker in tickers_gr:
        dates=[]
        open=[]
        high=[]
        low=[]
        close=[]
        volume=[]
        url='https://www.naftemporiki.gr/finance/Data/getHistoryData.aspx?symbol={}&type=csv'.format(ticker)
        with closing(req.get(url, verify=True, stream=True)) as r:
            f = (line.decode('utf-8') for line in r.iter_lines())
            reader = csv.reader(f, delimiter=';')
            for row in reader:
                dates.append(row[0])
                row[1]=row[1].replace(',','.')
                open.append(row[1])
                row[2]=row[2].replace(',','.')
                high.append(row[2])
                row[3]=row[3].replace(',','.')
                low.append(row[3])
                row[4]=row[4].replace(',','.')
                close.append(row[4])
                row[5]=row[5].replace(',','.')
                volume.append(row[5])
        del dates[0]
        del open[0]
        del high[0]
        del low[0]
        del close[0]
        del volume[0]
        df_temp=pd.DataFrame({'date':dates, 'open':open, 'high':high,'low':low,'close':close,'volume':volume})
        df_temp.iloc[:,1]=df_temp.iloc[:,1].astype(float)
        df_temp['date'] =pd.to_datetime(df_temp['date'],format="%d/%m/%Y")
        df_temp.iloc[:,1:]=df_temp.iloc[:,1:].astype(float)
        data=df_temp.reset_index(drop=True)#
        #print(ticker,len(data))
        greekstocks_data[ticker]=data
        close_data=greekstocks_data[ticker]['close']
    return greekstocks_data


st.set_page_config(layout="wide")
st.markdown('<link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css" integrity="sha384-Gn5384xqQ1aoWXA+058RXPxPg6fy4IWvTNh0E263XmFcJlSAwiGgFAW/dAiS6JXm" crossorigin="anonymous">', unsafe_allow_html=True)
st.markdown("""
<nav class="navbar fixed-top navbar-expand-lg navbar-dark" style="background-color: #CE2B08;">
  <a class="navbar-brand" target="_blank">GetYour.Portfolio@gmail.com</a>
  <button class="navbar-toggler" type="button" data-toggle="collapse" data-target="#navbarNav" aria-controls="navbarNav" aria-expanded="false" aria-label="Toggle navigation">
    <span class="navbar-toggler-icon"></span>
  </button>
  <div class="collapse navbar-collapse" id="navbarNav">
    <ul class="navbar-nav">
      <li class="nav-item active">
        <a class="nav-link disabled" href="#">Home <span class="sr-only">(current)</span></a>
      </li>
      <li class="nav-item">
        <a class="nav-link" target="_blank">Λίγη Θεωρία</a>
      </li>
      <li class="nav-item">
        <a class="nav-link"  target="_blank">Λίγη Πράξη</a>
      </li>
    </ul>
  </div>
</nav>
""", unsafe_allow_html=True)

st.markdown(''' 
#                       **Βελτιστοποιημένο Χαρτοφυλάκιο Μετοχών του ΧΑ**

### ΠΡΟΣΟΧΗ ότι βλέπετε εδώ είναι φτιαγμένο για ενημερωτικούς και εκπαιδευτικούς σκοπούς μόνο και σε καμιά περίπτωση δεν αποτελεί επενδυτική ή άλλου είδους πρόταση.
### Οι επενδύσεις σε μετοχές ενέχουν οικονομικό ρίσκο και ο δημιουργός της εφαρμογής δεν φέρει καμιά ευθύνη σε περίπτωση απώλειας περιουσίας.
#### Μπορείτε να επικοινωνείτε τα σχόλια και παρατηρήσεις σας με email στο <getyour.portfolio@gmail.com>.
''')

data_load_state = st.text('Loading data...')
# Load rows of data into a dataframe.
data = load_data(stocks)
# create the closing prices dataframe
l_close=pd.DataFrame(columns=['stock','date','lastprice','len_prices'])
close_data=pd.DataFrame()
i=1
for ticker in stocks:
    last_close=data[ticker].iloc[-1]['close']
    last_date=data[ticker].iloc[-1]['date']
    len_values=len(data[ticker])
    l_close=l_close.append({'stock':ticker,'date':last_date,'lastprice':last_close,
                            'len_prices':len_values},ignore_index=True)
    df_temp=data[ticker].loc[:,['date','close']].rename(columns={'close':ticker}).set_index('date')
    if i==1:
        close_data=df_temp
        i=i+1
    else:
        close_data=close_data.merge(df_temp,how='inner',on='date')

l_close_min=l_close['len_prices'].min()

st.write('Υπολογισμός βέλτιστου χαρτοφυλακίου από '+str(len(data))+' επιλεγμένες μετοχές του ΧΑ, βασισμένο στις αρχές της Σύγχρονης Θεωρίας Χαρτοφυλακίου.')
st.write('Στον παρακάτω πίνακα φαίνεται η λίστα των μετοχών με τις ημερήσιες τιμές κλεισίματός τους για τις τελευταίες '+str(l_close_min)+' μέρες')
st.write('Οι μετοχές που έχουν αρχικά επιλεγεί είναι οι παρακάτω που βλέπουμε στον πίνακα των τιμών κλεισίματος τους. Τα ονόματα τους είναι τα ονόματα των στηλών του πίνακα.')
st.write('Επιλέξτε από την στήλη αριστερά το μέγεθος, το βάθος χρόνου και το είδος του Χαρτοφυλακίου που θέλετε να ψάξει και να φτιάξει για σας η εφαρμογή.')
best_res=pd.DataFrame(columns=['trades', 'momentum_window', 'minimum_momentum', 'portfolio_size',
                              'tr_period', 'cutoff', 'tot_contribution', 'final port_value',
                              'cumprod', 'tot_ret', 'drawdown'])
df=close_data
q=st.sidebar.slider('Υπολογισμός με βάση τις τιμές των τελευταίων Χ ημερών',600, 1000, 700,50)
df_tr=df.tail(q)
df_pct=df_tr.pct_change()
df_cum_ret=pd.DataFrame()
for stock in stocks:
    df_cum_ret[stock]=cumulative_returns(stock, df_pct[stock])

st.write('Συσσωρευτικές αποδόσεις των παραπάνω μετοχών για τις Χ τελευταίες ημέρες, όπου Χ η επιλογή στην αριστερή στήλη.')
df_cum_ret=100*(df_cum_ret.iloc[-1:,:]-1)
st.dataframe(df_cum_ret.tail(10))
st.write('Πίνακας των ημερησίων ποσοστιαίων μεταβολών όλων των Μετοχών για τις Χ ημέρες')
st.dataframe(100*df_pct.tail(10))
corr_table = df_pct.corr()
corr_table['stock1'] = corr_table.index
corr_table = corr_table.melt(id_vars = 'stock1', var_name = 'stock2').reset_index(drop = True)
corr_table = corr_table[corr_table['stock1'] < corr_table['stock2']].dropna()
corr_table['abs_value'] = np.abs(corr_table['value'])
st.write('Πίνακας των τιμών των Συντελεστών Συσχέτισης των Μετοχών')
st.dataframe(corr_table)
st.write('Πάτα το παρακάτω κουμπί για να τρέξει το backtest με όλους τους συνδυασμούς των παραμέτρων')
if st.button('BACKTEST',key=2):
    #run backtest"
    port_value=10000
    new_port_value=0
    df=close_data
    #momentum_window = v
    #minimum_momentum = 70
    #portfolio_size=15
    #cutoff=0.1
    #added_value how much cash to add each trading period
    #tr_period=21 #trading period, 21 is a month,10 in a fortnite, 5 is a week, 1 is everyday
    #dataset=800 #start for length of days used for the optimising dataset
    #l_days=600  #how many days to use in optimisations
    res=pd.DataFrame(columns=['trades', 'momentum_window', 'minimum_momentum', 'portfolio_size',
                              'tr_period', 'cutoff', 'tot_contribution', 'final port_value',
                              'cumprod', 'tot_ret', 'drawdown'])
    #backtest 70%
    #x=int(len(df))
    #df_b=df .head(x) #backtest dataframe of first x values from total prices 
    #df_v=df.tail(len(df)-x) #validate dataframe of the rest prices
    #run all the combinations for all parameter values
    for momentum_window in [120,180,240,360]:
        for minimum_momentum in [70,100,120,150,180]:
            for portfolio_size in [5,10,15,20]:
                for tr_period in [5,10,20]:
                    for cutoff in [0.05,0.1]:
                        port_value=10000
                        l_days=700
                        dataset=800
                        added_value=0
                        rs=backtest_portfolio(df,dataset,l_days,momentum_window,minimum_momentum,portfolio_size,tr_period,cutoff,port_value,added_value)
                        res=res.append(rs, ignore_index=True)
                        #print(rs)
                        #print(res.sort_values(by=['tot_ret']).tail(2))
    #print(res.sort_values(by=['tot_ret']).tail(10))

    best_res=res.sort_values(by=['tot_ret']).tail(1).reset_index(drop=True)
    st.write('Ο καλύτερος συνδυασμός των παραμέτρων με τα μέχρι σήμερα ιστορικά στοιχεία είναι ο παρακάτω')
    st.write(best_res)

#-----Γενικές παράμετροι
st.sidebar.write('ΠΑΡΑΜΕΤΡΟΙ ΧΑΡΤΟΦΥΛΑΚΙΟΥ')
port_value=st.sidebar.slider('Αρχική επένδυση στο χαρτοφυλάκιο €', 1000, 10000, 5000,100)
cutoff=st.sidebar.slider('Ελάχιστο Ποσοστό Συμμετοχής μιας Μετοχής στο Χαρτοφυλάκιο.', 0.01, 0.20, 0.10, 0.01)
momentum_window=st.sidebar.slider('Πλήθος τιμών Μετοχής για τον υπολογισμό του momentum indicator.',90, 500, 120,10)
minimum_momentum=st.sidebar.slider('Ελάχιστη τιμή του momentum indicator μιας Μετοχής για να συμπεριληφθεί στο χαρτοφυλάκιο.',70, 180, 120,10)
portfolio_size=st.sidebar.slider('Μέγιστο Πλήθος Μετοχών που θα περιέχει το Χαρτοφυλάκιο.',5, 25, 10, 1)
df_m=pd.DataFrame()
m_s=[]
sto=[]
for s in stocks:
    sto.append(s)
    m_s.append(momentum_score(df_tr[s].tail(momentum_window)))
df_m['stock']=sto
df_m['momentum']=m_s
dev=df_m['momentum'].std()
# Get the top momentum stocks for the period
df_m = df_m.sort_values(by='momentum', ascending=False)
df_m=df_m[(df_m['momentum']>minimum_momentum-0.5*dev)&(df_m['momentum']<minimum_momentum+1.9*dev)].head(portfolio_size)
# Set the universe to the top momentum stocks for the period
universe = df_m['stock'].tolist()

# Create a df with just the stocks from the universe
df_t = select_columns(df_tr, universe)
st.write(df_t.tail())
#-----Χαρτοφυλάκιο Νο1 γενικό
#Calculate portofolio mu and S
mu =capm_returns(df_t)
S = CovarianceShrinkage(df_t).ledoit_wolf()
# Optimise the portfolio for maximal Sharpe ratio
ef = EfficientFrontier(mu, S) # Use regularization (gamma=1)
weights=ef.min_volatility()
cleaned_weights = ef.clean_weights(cutoff=cutoff,rounding=3)
ef.portfolio_performance()

st.subheader('Βελτιστοποιημένο Χαρτοφυλάκιο')
st.write('Το προτεινόμενο χαρτοφυλάκιο από τις ιστορικές τιμές των επιλεγμένων μετοχών έχει τα παρακάτω χαρακτηριστικά')
st.write('Αρχική Αξία Χαρτοφυλακίου : '+str(port_value)+'€')
st.write('Sharpe Ratio: '+str(round(ef.portfolio_performance()[2],2)))
st.write('Απόδοση Χαρτοφυλακίο: '+str(round(ef.portfolio_performance()[0]*100,2))+'%')
st.write('Μεταβλητότητα Χαρτοφυλακίου: '+str(round(ef.portfolio_performance()[1]*100,2))+'%')
# Allocate
latest_prices = get_latest_prices(df_t)
da =DiscreteAllocation( cleaned_weights,
                        latest_prices,
                        total_portfolio_value=port_value
                        )
allocation = da.greedy_portfolio()[0]
non_trading_cash=da.greedy_portfolio()[1]

# Put the stocks and the number of shares from the portfolio into a df
symbol_list = []
mom=[]
w=[]
num_shares_list = []
l_price=[]
tot_cash=[]
for symbol, num_shares in allocation.items():
    symbol_list.append(symbol)
    w.append(cleaned_weights[symbol])
    num_shares_list.append(num_shares)
    l_price.append(latest_prices[symbol])
    tot_cash.append(num_shares*latest_prices[symbol])
    
    
df_buy=pd.DataFrame()
df_buy['stock']=symbol_list
df_buy['weights']=w
df_buy['shares']=num_shares_list
df_buy['price']=l_price
df_buy['value']=tot_cash
st.write('Επενδυμένο σε μετοχές {0:.2f}€ ή το {1:.2f}% του χαρτοφυλακίου'.format(df_buy['value'].sum(),100*df_buy['value'].sum()/port_value))
st.write('Εναπομείναντα μετρητά :{0:.2f}€ ή το {1:.2f}% του χαρτοφυλακίου'.format(port_value-df_buy['value'].sum(),100-100*df_buy['value'].sum()/port_value))
df_buy=df_buy.append({'stock':'CASH','weights': round(1-df_buy['value'].sum()/port_value,2),'shares':1,'price':round(port_value-df_buy['value'].sum(),2),'value':round(port_value-df_buy['value'].sum(),2)}, ignore_index=True)
#df_buy=df_buy.set_index('stock')
st.dataframe(df_buy)
 
rs=backtest_portfolio(df,dataset=800,l_days=700,momentum_window=momentum_window,minimum_momentum=minimum_momentum,portfolio_size=portfolio_size,tr_period=20,cutoff=cutoff,port_value=port_value,added_value=0)
st.dataframe(pd.DataFrame(rs))

st.write('Στον παραπάνω πίνακα βλέπουμε το σύμβολο της κάθε μετοχής, στην στήλη "weights" το ποσοστό συμμετοχής της στο χαρτοφυλάκιο,')
st.write('στην στήλη "shares" το πλήθος των μετοχών, στην στήλη "price" την τιμή αγοράς της κάθε μετοχής και')  
st.write('στην στήλη "value" το συνολικό ποσό χρημάτων που επενδύεται στην κάθε μετοχή')
st.write('Εάν θέλεις να σώσεις το παραπάνω χαρτοφυλάκιο τότε δώσε ένα όνομα και ένα email και μετά πάτησε το κουμπί για να σου αποσταλεί σαν αρχείο.')
filenm=st.text_input('Δώσε ένα όνομα στο Χαρτοφυλάκιο', value="My Portfolio",key=1)
if st.button('Σώσε αυτό το Χαρτοφυλάκιο τύπου 1',key=1):
    filename = filenm+'.csv'
    download_button_str = download_button(df_buy, filename, f'Click here to download {filename}', pickle_it=False)
    st.markdown(download_button_str, unsafe_allow_html=True)
    
st.subheader('Εαν έχετε προηγουμένως χρησιμοποιήσει την εφαρμογή και έχετε ζητήσει ένα Χαροφυλάκιο ανεβάστε το csv αρχείο στο παρακάτω πεδίο για να δείτε την απόδοσή του σήμερα.')    
st.markdown(STYLE, unsafe_allow_html=True)
file = st.file_uploader("Σύρτε και αφήστε εδώ το Χαρτοφυλάκιό σας έφτιαξε η εφαρμογή για σας (*.csv)", type='csv')
show_file = st.empty()
if not file:
    show_file.info("")
else:    
    df_old = pd.read_csv(file)
    df_old=df_old.rename(columns={'price':'bought price'})
    last_price=[]
    new_values=[]
    new_weights=[]
    pct=[]
    for stock in list(df_old.iloc[:-1]['stock']):
        last_price.append(df.iloc[-1][stock])
        nv=df_old.loc[df_old['stock']==stock,'shares'].values[0]*df.iloc[-1][stock]
        new_values.append(nv)
        pt=round(100*(df.iloc[-1][stock]/df_old.loc[df_old['stock']==stock,'bought price'].values[0]-1),2)
        pct.append(pt)
    last_price.append(0)
    pct.append(0)
    df_old['last price']=last_price
    new_values.append(df_old.iloc[-1]['value'])
    df_old['new value']=new_values
    df_old['pct_change%']=pct
    new_port_value=df_old['new value'].sum()
    for stock in list(df_old.iloc[:-1]['stock']):
        new_weights.append(df_old.loc[df_old['stock']==stock,'shares'].values[0]*df.iloc[-1][stock]/new_port_value)
    new_weights.append(df_old.iloc[-1]['new value']/ new_port_value)
    df_old['new weights']=new_weights    
    st.write(f'Αρχική αξία του Χαροφυλακίου ήταν :{df_old["value"].sum()} €')
    st.write(f'Τώρα είναι : {round(new_port_value,2)} €')
    st.write(f'δηλ. έχουμε μια απόδοση ίση με {100*round(new_port_value/df_old["value"].sum()-1,4)} %')
    st.dataframe(df_old)
    file.close()
    rebalance_portfolio(df_old,df_buy)
                          
