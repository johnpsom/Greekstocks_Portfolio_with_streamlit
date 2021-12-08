# -*- coding: utf-8 -*-
"""
Created on Sun Oct 25 13:19:50 2020
@author: getyour.portfolio@gmail.com
"""
import streamlit as st
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

import smtplib, ssl

import io
import base64
import os
import json
import pickle
import uuid
import re

from email import encoders
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.utils import parseaddr
from enum import Enum
from io import BytesIO, StringIO
from typing import Union

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

#send your portfolio by email
def send_portfolio_byemail(filename, receiver_email):
    smtp_server = "smtp.gmail.com"
    port =465 # For starttls
    gmail_user = st.secrets["sender_email"]
    gmail_password = st.secrets["password"]
    subject = "Το Χαρτοφυλάκιό σου"
    body = "Λαμβάνεις αυτό το μύνημα διότι έφτιαξες ένα χαρτοφυλάκιο και ζήτησες να σου αποσταλεί. Βρες στο συνημμένο αρχείο το χαρτοφυλάκιο που έχεις φτιάξει."
    # Create a multipart message and set headers
    message = MIMEMultipart()
    message["From"] = gmail_user
    message["To"] = receiver_email
    message["Subject"] = subject
    # Add body to email
    message.attach(MIMEText(body, "plain"))
    # Open file=filename in binary mode
    with open(filename, "rb") as attachment:
        # Add file as application/octet-stream
        # Email client can usually download this automatically as attachment
        part = MIMEBase("application", "octet-stream")
        part.set_payload(attachment.read())
    # Encode file in ASCII characters to send by email    
    encoders.encode_base64(part)
    # Add header as key/value pair to attachment part
    part.add_header("Content-Disposition",f"attachment; filename= {filename}")
    # Add attachment to message and convert message to string
    message.attach(part)
    email_text = message.as_string()
    #Create SMTP session for sending the mail
    server = smtplib.SMTP('smtp.gmail.com:587')
    server.ehlo_or_helo_if_needed()
    server.starttls()
    server.ehlo_or_helo_if_needed()
    session.login(gmail_user, gmail_password) #login with mail_id and password
    server.sendmail(gmail_user, receiver_email , email_text)
    session.quit()
    st.write('Αποστολή email OK!')
    
    return        

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


#stock universe 
stocks=['AEGN.ATH', 'AETF.ATH', 'ALMY.ATH', 'ALPHA.ATH', 'ANDRO.ATH', 'ANEK.ATH', 'ASCO.ATH',
     'ASTAK.ATH', 'ATEK.ATH', 'ATRUST.ENAX', 'ATTICA.ATH', 'AVAX.ATH', 'AVE.ATH', 'BELA.ATH', 'BIOKA.ATH',
     'BIOSK.ATH', 'BIOT.ATH', 'BRIQ.ATH', 'BYTE.ATH', 'CENER.ATH',  'CRETA.ATH', 'DAIOS.ATH', 'DOMIK.ATH',
     'DROME.ATH', 'DUR.ATH', 'EEE.ATH', 'EKTER.ATH', 'ELBE.ATH', 'ELBIO.ATH', 'ELHA.ATH', 'ELIN.ATH', 'ELLAKTOR.ATH',
     'ELPE.ATH', 'ELSTR.ATH', 'ELTON.ATH', 'ENTER.ATH', 'EPSIL.ATH', 'ETE.ATH', 'EUPIC.ATH', 'EUROB.ATH', 'EVROF.ATH',
     'EXAE.ATH', 'EYAPS.ATH', 'EYDAP.ATH', 'FIER.ATH', 'FLEXO.ATH', 'FOODL.ENAX', 'FOYRK.ATH', 'GEBKA.ATH', 'GEKTERNA.ATH',
     'HTO.ATH', 'IATR.ATH', 'IKTIN.ATH', 'ILYDA.ATH', 'INKAT.ATH', 'INLOT.ATH', 'INTERCO.ATH', 'INTET.ATH', 'INTRK.ATH',
     'KAMP.ATH', 'KEKR.ATH', 'KEPEN.ATH', 'KLM.ATH', 'KMOL.ATH', 'KORDE.ATH', 'KREKA.ATH', 'KRI.ATH', 'KTILA.ATH',
     'KYLO.ATH', 'KYRI.ATH', 'LAMDA.ATH', 'LANAC.ATH', 'LAVI.ATH', 'LEBEK.ATH', 'LOGISMOS.ATH', 'LYK.ATH', 'MATHIO.ATH', 'MEDIC.ATH',
     'MERKO.ATH', 'MEVA.ATH', 'MIG.ATH', 'MIN.ATH', 'MOH.ATH', 'MYTIL.ATH', 'OLTH.ATH', 'OLYMP.ATH', 'OPAP.ATH', 'OTOEL.ATH',
     'PAIR.ATH', 'PAP.ATH', 'PERF.ENAX', 'PETRO.ATH', 'PLAIS.ATH', 'PLAKR.ATH', 'PLAT.ATH', 'PPA.ATH',
     'PROF.ATH', 'QUAL.ATH', 'QUEST.ATH', 'REVOIL.ATH', 'SAR.ATH', 'SPACE.ATH', 'SPIR.ATH', 'TATT.ATH',
     'TELL.ATH', 'TENERGY.ATH',  'TPEIR.ATH', 'TRASTOR.ATH', 'VARG.ATH', 'VARNH.ATH', 'VIDAVO.ENAX', 'VIO.ATH',
     'VIS.ATH', 'VOSYS.ATH', 'YALCO.ATH','ADMIE.ATH','PPC.ATH']
st.set_page_config(layout="wide")
st.title('Βελτιστοποιημένο Χαρτοφυλάκιο Μετοχών του ΧΑ')
data_load_state = st.text('Loading data...')
# Load rows of data into the dataframe.
data = load_data(stocks)
# Notify the reader that the data was successfully loaded.
data_load_state.text("Done! (using st.cache)")
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

st.subheader('ΠΡΟΣΟΧΗ ότι βλέπετε εδώ είναι φτιαγμένο για ενημερωτικούς και εκπαιδευτικούς σκοπούς μόνο και σε καμιά περίπτωση δεν αποτελεί επενδυτική ή άλλου είδους πρόταση.')
st.subheader('Οι επενδύσεις σε μετοχές ενέχουν οικονομικό ρίσκο και ο δημιουργός της εφαρμογής δεν φέρει καμιά ευθύνη σε περίπτωση απώλειας περιουσίας.')
st.subheader('Μπορείτε να επικοινωνείτε τα σχόλια και παρατηρήσεις σας στο email: getyour.portfolio@gmail.com .')
st.write('Υπολογισμός βέλτιστου χαρτοφυλακίου από '+str(len(data))+' επιλεγμένες μετοχές του ΧΑ, βασισμένο στις αρχές της Σύγχρονης Θεωρίας Χαρτοφυλακίου του Νομπελίστα Οικονομολόγου Harry Markowitz.')
st.write('Στον παρακάτω πίνακα φαίνεται η λίστα των μετοχών με τις ημερήσιες τιμές κλεισίματός τους για τις τελευταίες '+str(l_close_min)+' μέρες')
st.write('Οι μετοχές που έχουν αρχικά επιλεγεί είναι οι παρακάτω που βλέπουμε στον πίνακα των τιμών κλεισίματος τους. Τα ονόματα τους είναι τα ονόματα των στηλών του πίνακα.')
st.write('Επιλέξτε από την στήλη αριστερά το μέγεθος, το βάθος χρόνου και το είδος του Χαρτοφυλακίου που θέλετε να ψάξει και να φτιάξει για σας η εφαρμογή.')


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
#-----Γενικές παράμετροι
st.sidebar.write('ΠΑΡΑΜΕΤΡΟΙ ΒΕΛΤΙΣΤΟΠΟΙΗΜΕΝΩΝ ΧΑΡΤΟΦΥΛΑΚΙΩΝ')
port_value=st.sidebar.slider('Αρχική επένδυση στο χαρτοφυλάκιο €', 1000, 10000, 5000,100)
cutoff=st.sidebar.slider('Ελάχιστο Ποσοστό Συμμετοχής μιας Μετοχής στο Χαρτοφυλάκιο.', 0.01, 0.20, 0.10, 0.01)
momentum_window=st.sidebar.slider('πλήθος τιμών Μετοχής στον υπολογισμό του momentum indicator.',90, 500, 120,10)
minimum_momentum=st.sidebar.slider('Ελάχιστο τιμή του momentum indicator.',70, 180, 120,10)
portfolio_size=st.sidebar.slider('Μέγιστο Πλήθος Μετοχών.',5, 20, 10, 1)
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
df_buy=df_buy.set_index('stock')
st.dataframe(df_buy)
st.write('Στον παραπάνω πίνακα βλέπουμε το σύμβολο της κάθε μετοχής, στην στήλη "weights" το ποσοστό συμμετοχής της στο χαρτοφυλάκιο,')
st.write('στην στήλη "shares" το πλήθος των μετοχών, στην στήλη "price" την τιμή αγοράς της κάθε μετοχής και')  
st.write('στην στήλη "value" το συνολικό ποσό χρημάτων που επενδύεται στην κάθε μετοχή')
st.write('Εάν θέλεις να σώσεις το παραπάνω χαρτοφυλάκιο τότε δώσε ένα όνομα και ένα email και μετά πάτησε το κουμπί για να σου αποσταλεί σαν αρχείο.')
filenm=st.text_input('Δώσε ένα όνομα στο Χαρτοφυλάκιο', value="My Portfolio",key=1)
#receiver_email=st.text_input('Ποιό είναι το email στο οποίο θα αποσταλεί το χαρτοφυλάκιο?',value='example@example.com',key=2)
if st.button('Σώσε αυτό το Χαρτοφυλάκιο τύπου 1',key=1):
    filename = filenm+'.csv'
    download_button_str = download_button(df_buy, filename, f'Click here to download {filename}', pickle_it=False)
    st.markdown(download_button_str, unsafe_allow_html=True)
    
    
    
    
    
    #if '@' in parseaddr(receiver_email)[1]:
    #    send_portfolio_byemail(filenm,receiver_email)



st.subheader('Εαν έχετε προηγουμένως χρησιμοποιήσει την εφαρμογή και έχετε ζητήσει ένα Χαροφυλάκιο ανεβάστε το csv αρχείο στο παρακάτω πεδίο για να δείτε την απόδοσή του σήμερα.')    
st.markdown(STYLE, unsafe_allow_html=True)
file = st.file_uploader("Σύρτε και αφήστε εδώ το Χαρτοφυλάκιό σας έφτιαξε η εφαρμογή για σας (*.csv)", type='csv')
show_file = st.empty()
if not file:
    show_file.info("")
else:    
    df1 = pd.read_csv(file)
    df1=df1.iloc[:,1:]
    print(df1)
    df1=df1.rename(columns={'price':'bought price'})
    last_price=[]
    new_values=[]
    new_weights=[]
    pct=[]
    for stock in list(df1.iloc[:-1]['stock']):
        last_price.append(df.iloc[-1][stock])
        nv=df1.loc[df1['stock']==stock,'shares'].values[0]*df.iloc[-1][stock]
        new_values.append(nv)
        pt=round(100*(df.iloc[-1][stock]/df1.loc[df1['stock']==stock,'bought price'].values[0]-1),2)
        pct.append(pt)
    last_price.append(0)
    pct.append(0)
    df1['last price']=last_price
    new_values.append(df1.iloc[-1]['value'])
    df1['new value']=new_values
    df1['pct_change%']=pct
    new_port_value=df1['new value'].sum()
    for stock in list(df1.iloc[:-1]['stock']):
        new_weights.append(df1.loc[df1['stock']==stock,'shares'].values[0]*df.iloc[-1][stock]/new_port_value)
    new_weights.append(df1.iloc[-1]['new value']/ new_port_value)
    df1['new weights']=new_weights    
    st.write('Αρχική αξία του Χαροφυλακίου ήταν :'+str(df1['value'].sum())+'€')
    st.write('Τώρα είναι :'+str(round(new_port_value,2))+'€')
    st.write(' δηλ. έχουμε μια απόδοση ίση με '+str(100*round(new_port_value/df1['value'].sum()-1,4))+'%')
    st.dataframe(df1)
    file.close()
    
