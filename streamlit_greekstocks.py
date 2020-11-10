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
import os
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
def load_data():
    df=pd.DataFrame()
    stocks=['CENER.ATH','CNLCAP.ATH','TITC.ATH','AVAX.ATH','AVE.ATH','ADMIE.ATH','ALMY.ATH','ALPHA.ATH','AEGN.ATH',
            'ASCO.ATH','TATT.ATH','VIO.ATH','BIOSK.ATH','VOSYS.ATH','BYTE.ATH','GEBKA.ATH','GEKTERNA.ATH','PPC.ATH',
            'DOMIK.ATH','EEE.ATH','EKTER.ATH','ELIN.ATH','TELL.ATH','ELLAKTOR.ATH','ELPE.ATH','ELTON.ATH','ELHA.ATH','ENTER.ATH',
            'EPSIL.ATH','EYAPS.ATH','ETE.ATH','EYDAP.ATH','EUPIC.ATH','EUROB.ATH','EXAE.ATH','IATR.ATH','IKTIN.ATH','ILYDA.ATH',
            'INKAT.ATH','INLOT.ATH','INTERCO.ATH','INTET.ATH','INTRK.ATH','KAMP.ATH','KEKR.ATH','KEPEN.ATH',
            'KLM.ATH','KMOL.ATH','QUAL.ATH','QUEST.ATH','KRI.ATH','LAVI.ATH','LAMDA.ATH','KYLO.ATH','LYK.ATH','MEVA.ATH',
            'MERKO.ATH','MIG.ATH','MIN.ATH','MOH.ATH','BELA.ATH','BRIQ.ATH','MYTIL.ATH','NEWS.ATH','OLTH.ATH','PPA.ATH',
            'OLYMP.ATH','OPAP.ATH','HTO.ATH','OTOEL.ATH','PAIR.ATH','PAP.ATH','PASAL.ATH','TPEIR.ATH','PERF.ENAX',
            'PETRO.ATH','PLAT.ATH','PLAIS.ATH','PLAKR.ATH','PPAK.ATH','PROF.ATH','REVOIL.ATH','SAR.ATH','SPACE.ATH',
            'SPIR.ATH','TENERGY.ATH','TRASTOR.ATH','FLEXO.ATH','FOYRK.ATH','FORTH.ATH'           
            ]
    i=1
    for stock in stocks:
        dates=[]
        close=[]
        url='https://www.naftemporiki.gr/finance/Data/getHistoryData.aspx?symbol={}&type=csv'.format(stock)
        with closing(req.get(url, verify=True, stream=True)) as r:
            f = (line.decode('utf-8') for line in r.iter_lines())
            reader = csv.reader(f, delimiter=';')
            for row in reader:
                dates.append(row[0])
                row[4]=row[4].replace(',','.')
                close.append(row[4])
                
        del dates[0]
        del close[0]
        df_temp=pd.DataFrame({'Dates':dates, stock:close})
        #df_temp=df_temp.apply(lambda x: x.str.replace(',','.'))
        df_temp=df_temp.tail(600)
        if i>1:
            df=df.join(df_temp.set_index('Dates'), on='Dates',how='inner')
        if i==1:
            df=df_temp
            i=i+1
           
    a=df['Dates']
    df=df.iloc[:,1:].astype('float')
    df.insert(0,'Dates', a)
    df.to_csv('greek_stockdata.csv')
    df=df.reset_index()
    df=df.set_index('Dates')
    return df

# Momentum score function
def momentum_score(ts):
    x = np.arange(len(ts))
    log_ts = np.log(ts)
    regress = stats.linregress(x, log_ts)
    annualized_slope = (np.power(np.exp(regress[0]), 252) -1) * 100
    return annualized_slope * (regress[2] ** 2)

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
    sender_email = "getyour.portfolio@gmail.com"
    password = MY_SECRET_NAME
    subject = "Το Χαρτοφυλάκιό σου"
    body = "Βρες στο συνημμένο αρχείο το χαρτοφυλάκιο που έχεις φτιάξει."
    # Create a multipart message and set headers
    message = MIMEMultipart()
    message["From"] = sender_email
    message["To"] = receiver_email
    message["Subject"] = subject
    message["Bcc"] = receiver_email  # Recommended for mass emails
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
    part.add_header("Content-Disposition",
        f"attachment; filename= {filename}",
    )
    
    # Add attachment to message and convert message to string
    message.attach(part)
    text = message.as_string()
    
    # Log in to server using secure context and send email
    context = ssl.create_default_context()
    with smtplib.SMTP_SSL(smtp_server, port, context=context) as server:
        server.login(sender_email, password)
        server.sendmail(sender_email, receiver_email, text)
        #after send email delete the csv file
        if os.path.isfile(filename):
            os.remove(filename) 
    return        

#stock universe 
stocks=['CENER.ATH','CNLCAP.ATH','TITC.ATH','AVAX.ATH','AVE.ATH','ADMIE.ATH','ALMY.ATH','ALPHA.ATH','AEGN.ATH',
            'ASCO.ATH','TATT.ATH','VIO.ATH','BIOSK.ATH','VOSYS.ATH','BYTE.ATH','GEBKA.ATH','GEKTERNA.ATH','PPC.ATH',
            'DOMIK.ATH','EEE.ATH','EKTER.ATH','ELIN.ATH','TELL.ATH','ELLAKTOR.ATH','ELPE.ATH','ELTON.ATH','ELHA.ATH','ENTER.ATH',
            'EPSIL.ATH','EYAPS.ATH','ETE.ATH','EYDAP.ATH','EUPIC.ATH','EUROB.ATH','EXAE.ATH','IATR.ATH','IKTIN.ATH','ILYDA.ATH',
            'INKAT.ATH','INLOT.ATH','INTERCO.ATH','INTET.ATH','INTRK.ATH','KAMP.ATH','KEKR.ATH','KEPEN.ATH',
            'KLM.ATH','KMOL.ATH','QUAL.ATH','QUEST.ATH','KRI.ATH','LAVI.ATH','LAMDA.ATH','KYLO.ATH','LYK.ATH','MEVA.ATH',
            'MERKO.ATH','MIG.ATH','MIN.ATH','MOH.ATH','BELA.ATH','BRIQ.ATH','MYTIL.ATH','NEWS.ATH','OLTH.ATH','PPA.ATH',
            'OLYMP.ATH','OPAP.ATH','HTO.ATH','OTOEL.ATH','PAIR.ATH','PAP.ATH','PASAL.ATH','TPEIR.ATH','PERF.ENAX',
            'PETRO.ATH','PLAT.ATH','PLAIS.ATH','PLAKR.ATH','PPAK.ATH','PROF.ATH','REVOIL.ATH','SAR.ATH','SPACE.ATH',
            'SPIR.ATH','TENERGY.ATH','TRASTOR.ATH','FLEXO.ATH','FOYRK.ATH','FORTH.ATH'           
            ]
st.set_page_config(layout="wide")
st.title('Βελτιστοποιημένο Χαρτοφυλάκιο Μετοχών του ΧΑ')
data_load_state = st.text('Loading data...')
# Load rows of data into the dataframe.
data = load_data()
# Notify the reader that the data was successfully loaded.
data_load_state.text("Done! (using st.cache)")
st.subheader('ΠΡΟΣΟΧΗ ότι βλέπετε εδώ είναι φτιαγμένο για ενημερωτικούς και εκπαιδευτικούς σκοπούς μόνο και σε καμιά περίπτωση δεν αποτελεί επενδυτική ή άλλου είδους πρόταση.')
st.subheader('Οι επενδύσεις σε μετοχές ενέχουν οικονομικό ρίσκο και ο δημιουργός της εφαρμογής δεν φέρει καμιά ευθύνη σε περίπτωση απώλειας περιουσίας.')
st.subheader('Μπορείτε να επικοινωνείτε τα σχόλια και παρατηρήσεις σας στο email: getyour.portfolio@gmail.com .')
st.write('Υπολογισμός βέλτιστου χαρτοφυλακίου από 90 επιλεγμένες μετοχές του ΧΑ, βασισμένο στις αρχές της Σύγχρονης Θεωρίας Χαρτοφυλακίου του Νομπελίστα Οικονομολόγου Harry Markowitz.')
st.write('Στον παρακάτω πίνακα φαίνεται η λίστα των μετοχών με τις ημερήσιες τιμές κλεισίματός τους για τις τελευταίες '+str(len(data))+' μέρες')
st.write('Οι μετοχές που έχουν αρχικά επιλεγεί είναι οι παρακάτω που βλέπουμε στον πίνακα των τιμών κλεισίματος τους. Τα ονόματα τους είναι τα ονόματα των στηλών του πίνακα.')
st.write('Επιλέξτε από την στήλη αριστερά το μέγεθος, το βάθος χρόνου και το είδος του Χαρτοφυλακίου που θέλετε να ψάξει και να φτιάξει για σας η εφαρμογή.')

st.dataframe(data=data.iloc[:,1:])
df=data.iloc[:,1:]
q=st.sidebar.slider('Υπολογισμός με βάση τις τιμές των τελευταίων Χ ημερών', 60, 300, 180,10)
df_t=df.tail(q)
df_pct=df_t.pct_change()
df_cum_ret=pd.DataFrame()
for stock in stocks:
    df_cum_ret[stock]=cumulative_returns(stock, df_pct[stock])

st.write('Συσσωρευτικές αποδόσεις των παραπάνω μετοχών για τις Χ τελευταίες ημέρες, όπου Χ η επιλογή στην αριστερή στήλη.')
st.dataframe(100*(df_cum_ret.iloc[-1:,:]-1))
m_cum_ret=pd.DataFrame((df_cum_ret.iloc[-1:,:])).max()
max_ret=round(100*(m_cum_ret.max()-1),0)
st.write('Πίνακας των ημερησίων ποσοστιαίων μεταβολών όλων των Μετοχών για τις Χ ημέρες')
st.dataframe(df_pct)
corr_table = df_pct.corr()
corr_table['stock1'] = corr_table.index
corr_table = corr_table.melt(id_vars = 'stock1', var_name = 'stock2').reset_index(drop = True)
corr_table = corr_table[corr_table['stock1'] < corr_table['stock2']].dropna()
corr_table['abs_value'] = np.abs(corr_table['value'])
st.write('Πίνακας των τιμών των Συντελεστών Συσχέτισης των Μετοχών')
st.dataframe(corr_table)
#-----Γενικές παράμετροι
st.sidebar.write('ΠΑΡΑΜΕΤΡΟΙ ΒΕΛΤΙΣΤΟΠΟΙΗΜΕΝΩΝ ΧΑΡΤΟΦΥΛΑΚΙΩΝ')
port_value=st.sidebar.slider('Αρχική επένδυση στο χαρτοφυλάκιο €', 1000, 10000, 3000,1000)
riskmo = st.sidebar.checkbox('Επιλeγμένο επιλέγει το μοντέλο ρίσκου Ledoit Wolf αλλιώς χρησιμοποιεί τον πίνακα των συνδιακυμάνσεων των Μετοχών.')
weightsmo=st.sidebar.checkbox('Επιλεγμένο επιλέγει τον υπολογισμό των βαρών με βάση τον μέγιστο Sharpe Ratio αλλιώς με την ελάχιστη διακύμανση.')
allocmo=st.sidebar.checkbox('Επιλεγμένο επιλέγει τον υπολογισμό του μοντέλου του greedy_portfolio αλλιώς επιλέγει το lp_portfolio.')
cutoff=st.sidebar.slider('Ελάχιστο Ποσοστό Συμμετοχής μιας Μετοχής στο Χαρτοφυλάκιο.', 0.01, 0.30, 0.05)

c1,c2,c3,c4= st.beta_columns((1,1,1,1))
#-----Χαρτοφυλάκιο Νο1 γενικό
#Calculate portofolio mu and S
mu = expected_returns.mean_historical_return(df_t)
if riskmo:
    S = CovarianceShrinkage(df_t).ledoit_wolf()
else:
    S = risk_models.sample_cov(df_t)
# Optimise the portfolio 
ef = EfficientFrontier(mu, S, gamma=2) # Use regularization (gamma=1)
if weightsmo:
    weights = ef.max_sharpe()
else:
    weights = ef.min_volatility()
cleaned_weights = ef.clean_weights(cutoff=cutoff,rounding=3)
ef.portfolio_performance()

c1.subheader('Χαρτοφυλάκιο Νο1')
c1.write('Το προτινόμενο χαρτοφυλάκιο από τις ιστορικές τιμές των επιλεγμένων μετοχών έχει τα παρακάτω χαρακτηριστικά')
c1.write('Αρχική Αξία Χαρτοφυλακίου : '+str(port_value)+'€')
c1.write('Sharpe Ratio: '+str(round(ef.portfolio_performance()[2],2)))
c1.write('Απόδοση Χαρτοφυλακίο: '+str(round(ef.portfolio_performance()[0]*100,2))+'%')
c1.write('Μεταβλητότητα Χαρτοφυλακίου: '+str(round(ef.portfolio_performance()[1]*100,2))+'%')
# Allocate
latest_prices = get_latest_prices(df_t)
da =DiscreteAllocation(
    cleaned_weights,
    latest_prices,
    total_portfolio_value=port_value
    )
if allocmo:
    allocation = da.greedy_portfolio()[0]
    non_trading_cash=da.greedy_portfolio()[1]
else:
    allocation = da.lp_portfolio()[0]
    non_trading_cash=da.lp_portfolio()[1]
# Put the stocks and the number of shares from the portfolio into a df
symbol_list = []
cw=[]
num_shares_list = []
l_price=[]
tot_cash=[]
for symbol, num_shares in allocation.items():
    symbol_list.append(symbol)
    cw.append(round(cleaned_weights[symbol],3))
    num_shares_list.append(num_shares)
    l_price.append(round(latest_prices[symbol],2))
    tot_cash.append(round(num_shares*latest_prices[symbol],2))
    
df_buy=pd.DataFrame()
df_buy['stock']=symbol_list
df_buy['weights']=cw
df_buy['shares']=num_shares_list
df_buy['price']=l_price
df_buy['value']=tot_cash

c1.write(df_buy)
c1.write('Επενδυμένο σε μετοχές {0:.2f}€ ή το {1:.2f}% του χαρτοφυλακίου'.format(df_buy['value'].sum(),100*df_buy['value'].sum()/port_value))
c1.write('Εναπομείναντα μετρητά :{0:.2f}€ ή το {1:.2f}% του χαρτοφυλακίου'.format(port_value-df_buy['value'].sum(),100-100*df_buy['value'].sum()/port_value))
df_buy=df_buy.append({'stock':'CASH','weights': round(1-df_buy['value'].sum()/port_value,2),'shares':0,'price':0,'value':round(port_value-df_buy['value'].sum(),2)}, ignore_index=True)
c1.write('Εάν θέλεις να σώσεις το παραπάνω χαρτοφυλάκιο τότε δώσε ένα όνομα και ένα email και μετά πάτησε το κουμπί για να σου αποσταλεί σαν αρχείο.')
filenm=c1.text_input('Δώσε ένα όνομα στο Χαρτοφυλάκιο', value="Portfolio1",key=1)
receiver_email=c1.text_input('Ποιό είναι το email στο οποίο θα αποσταλεί το χαρτοφυλάκιο?',value='example@example.com',key=2)
filenm=filenm+'.csv'
df_buy.to_csv(filenm)
if c1.button('Σώσε αυτό το Χαρτοφυλάκιο τύπου 1',key=1):
    if '@' in parseaddr(receiver_email)[1]:
        send_portfolio_byemail(filenm,receiver_email)
        
#------Χαρτοφυλάκιο Νο2
st.sidebar.write('Παράμετροι για το Χαρτοφυλάκιο Νο2')
c2.subheader('Χαρτοφυλάκιο Νο2')
c2.write('Θα γίνει ο ίδιος υπολογισμός αλλά ξεκινώντας από μικρότερο πλήθος μετοχών από τις αρχικές με βάση '+
         'τον υπολογισμό ενός δείκτη momentum και την κατάταξη των μετοχών σε φθίνουσα σειρά.'+
         ' Κατόπιν ο υπολογισμός του χαρτοφυλακίου έχει τις ίδιες επιλογές με το παραπάνω')
ps=st.sidebar.slider('Υπολογισμός με βάση αρχικά επιλεγμένο πλήθος μετοχών',5,30,10)
mom=st.sidebar.slider('Επιλογή με βάση την τιμή του mom',0,6,0)
portfolio_size=ps
#Calculate momentum and put the values in a dataframe
df_m=pd.DataFrame()
m_s=[]
stm=[]
for s in ['CENER.ATH','CNLCAP.ATH','TITC.ATH','AVAX.ATH','AVE.ATH','ADMIE.ATH','ALMY.ATH','ALPHA.ATH','AEGN.ATH',
            'ASCO.ATH','TATT.ATH','VIO.ATH','BIOSK.ATH','VOSYS.ATH','BYTE.ATH','GEBKA.ATH','GEKTERNA.ATH','PPC.ATH',
            'DOMIK.ATH','EEE.ATH','EKTER.ATH','ELIN.ATH','TELL.ATH','ELLAKTOR.ATH','ELPE.ATH','ELTON.ATH','ELHA.ATH','ENTER.ATH',
            'EPSIL.ATH','EYAPS.ATH','ETE.ATH','EYDAP.ATH','EUPIC.ATH','EUROB.ATH','EXAE.ATH','IATR.ATH','IKTIN.ATH','ILYDA.ATH',
            'INKAT.ATH','INLOT.ATH','INTERCO.ATH','INTET.ATH','INTRK.ATH','KAMP.ATH','KEKR.ATH','KEPEN.ATH',
            'KLM.ATH','KMOL.ATH','QUAL.ATH','QUEST.ATH','KRI.ATH','LAVI.ATH','LAMDA.ATH','KYLO.ATH','LYK.ATH','MEVA.ATH',
            'MERKO.ATH','MIG.ATH','MIN.ATH','MOH.ATH','BELA.ATH','BRIQ.ATH','MYTIL.ATH','NEWS.ATH','OLTH.ATH','PPA.ATH',
            'OLYMP.ATH','OPAP.ATH','HTO.ATH','OTOEL.ATH','PAIR.ATH','PAP.ATH','PASAL.ATH','TPEIR.ATH','PERF.ENAX',
            'PETRO.ATH','PLAT.ATH','PLAIS.ATH','PLAKR.ATH','PPAK.ATH','PROF.ATH','REVOIL.ATH','SAR.ATH','SPACE.ATH',
            'SPIR.ATH','TENERGY.ATH','TRASTOR.ATH','FLEXO.ATH','FOYRK.ATH','FORTH.ATH'           
            ]:
    stm.append(s)
    m_s.append(momentum_score(df_t[s]))
df_m['stock']=stm    
df_m['momentum'] = m_s
# Get the top momentum stocks for the period
df_m = df_m.sort_values(by='momentum', ascending=False).head(portfolio_size)
if mom==1:
    df_m=df_m[df_m['momentum']> df_m['momentum'].mean()-df_m['momentum'].std()]
if mom==2:
    df_m=df_m[df_m['momentum']> df_m['momentum'].mean()]
if mom==3:
    df_m=df_m[df_m['momentum']< df_m['momentum'].mean()+0.5*df_m['momentum'].std()]
if mom==4:
    df_m=df_m[df_m['momentum']< df_m['momentum'].mean()+df_m['momentum'].std()]    
if mom==5: 
    df_m=df_m[df_m['momentum']> 0]
if mom==6:
    df_m=df_m[df_m['momentum']< df_m['momentum'].mean()]
#print(df_m)
# Set the universe to the top momentum stocks for the period
universe = df_m['stock'].tolist()
# Create a df with just the stocks from the universe
df_tr = select_columns(df_t, universe)

#Calculate portofolio mu and S
mum = expected_returns.mean_historical_return(df_tr)
if riskmo:
    Sm = CovarianceShrinkage(df_tr).ledoit_wolf()
else:
    Sm = risk_models.sample_cov(df_tr)
# Optimise the portfolio 
efm = EfficientFrontier(mum, Sm, gamma=2) 

if weightsmo:
    weightsm = efm.max_sharpe()
else:
    #efm.add_objective(objective_functions.L2_reg, gamma=2)# Use regularization (gamma=1)
    weightsm = efm.min_volatility()
cleaned_weightsm = efm.clean_weights(cutoff=cutoff,rounding=3)
efm.portfolio_performance()
# Allocate
latest_pricesm = get_latest_prices(df_tr)
dam=DiscreteAllocation(
    cleaned_weightsm,
    latest_pricesm,
    total_portfolio_value=port_value
    )
if allocmo:
    allocationm = dam.greedy_portfolio()[0]
    non_trading_cashm=dam.greedy_portfolio()[1]
else:
    allocationm = dam.lp_portfolio()[0]
    non_trading_cashm=dam.lp_portfolio()[1]
# Put the stocks and the number of shares from the portfolio into a df
symbol_listm = []
mom=[]
cwm=[]
num_shares_listm = []
l_pricem=[]
tot_cashm=[]
for symbolm, num_sharesm in allocationm.items():
    symbol_listm.append(symbolm)
    cwm.append(round(cleaned_weightsm[symbolm],3))
    num_shares_listm.append(num_sharesm)
    l_pricem.append(round(latest_pricesm[symbolm],2))
    tot_cashm.append(round(num_sharesm*latest_pricesm[symbolm],2))
  
df_buym=pd.DataFrame()
df_buym['stock']=symbol_listm
df_buym['weights']=cwm
df_buym['shares']=num_shares_listm
df_buym['price']=l_pricem
df_buym['value']=tot_cashm

c2.write("Το προτινόμενο χαρτοφυλάκιο έχει τα παρακάτω χαρακτηριστικά")
c2.write('Αρχική Αξία Χαρτοφυλακίου : '+str(port_value)+'€')
c2.write('Sharpe Ratio: '+str(round(efm.portfolio_performance()[2],2)))
c2.write('Απόδοση Χαρτοφυλακίου: '+str(round(efm.portfolio_performance()[0]*100,2))+'%')
c2.write('Μεταβλητότητα Χαρτοφυλακίου: '+str(round(efm.portfolio_performance()[1]*100,2))+'%')
c2.write(df_buym)
c2.write('Επενδυμένο σε μετοχές {0:.2f}€ ή το {1:.2f}% του χαρτοφυλακίου'.format(df_buym['value'].sum(),100*df_buym['value'].sum()/port_value))
c2.write('Εναπομείναντα μετρητά :{0:.2f}€ ή το {1:.2f}% του χαρτοφυλακίου'.format(port_value-df_buym['value'].sum(),100-100*df_buym['value'].sum()/port_value))
df_buym=df_buym.append({'stock':'CASH','weights': round(1-df_buym['value'].sum()/port_value,2),'shares':0,'price':0,'value':round(port_value-df_buym['value'].sum(),2)}, ignore_index=True)
c2.write('Εάν θέλεις να σώσεις το παραπάνω χαρτοφυλάκιο τότε δώσε ένα όνομα και ένα email και μετά πάτησε το κουμπί για να σου αποσταλεί σαν αρχείο.')
filenm2=c2.text_input('Δώσε ένα όνομα στο Χαρτοφυλάκιο', value="Portfolio2",key=1)
receiver_email2=c2.text_input('Ποιό είναι το email στο οποίο θα αποσταλεί το χαρτοφυλάκιο?',value='example@example.com',key=4)
filenm2=filenm2+'.csv'
df_buym.to_csv(filenm2)
if c2.button('Σώσε αυτό το Χαρτοφυλάκιο τύπου 2',key=2):
    if '@' in parseaddr(receiver_email)[1]:
        send_portfolio_byemail(filenm2,receiver_email2)

#-----------------------------------------
st.sidebar.write('Παράμετροι για το Χαρτοφυλάκιο Νο3')
c3.subheader('Χαρτοφυλάκιο Νο3')
c3.write('Το προτεινόμενο χαρτοφυλάκιο με τις μετοχές ελάχιστης συσχέτισης έχει τα παρακάτω χαρακτηριστικά')
me=st.sidebar.slider('Από Πόσες μετοχές ελάχιστης συσχέτισης? ',10,50,20,1)
c3.write('Αρχική Αξία Χαρτοφυλακίου : '+str(port_value)+'€')
highest_corr = corr_table.sort_values("abs_value",ascending = True).head(me)
hi_corr_stocks_list=list(set(list(highest_corr['stock1'])) | set(highest_corr['stock2']))
hi_corr_universe = hi_corr_stocks_list
# Create a df with just the stocks from the universe
df_hicorr = select_columns(df_t, hi_corr_universe)
muh = expected_returns.mean_historical_return(df_hicorr)
if riskmo:
    Sh = CovarianceShrinkage(df_hicorr).ledoit_wolf()
else:
    Sh = risk_models.sample_cov(df_hicorr)
# Optimise the portfolio 
efh = EfficientFrontier(muh, Sh, gamma=2) # Use regularization (gamma=1)
if weightsmo:
    weightsh = efh.max_sharpe()
else:
    weightsh = efh.min_volatility()
cleaned_weightsh = efh.clean_weights(cutoff=cutoff,rounding=3)
efh.portfolio_performance()

c3.write('Sharpe Ratio: '+str(round(efh.portfolio_performance()[2],2)))
c3.write('Απόδοση Χαρτοφυλακίο: '+str(round(efh.portfolio_performance()[0]*100,2))+'%')
c3.write('Μεταβλητότητα Χαρτοφυλακίου: '+str(round(efh.portfolio_performance()[1]*100,2))+'%')
# Allocate
latest_pricesh = get_latest_prices(df_hicorr)
dah =DiscreteAllocation(
    cleaned_weightsh,
    latest_pricesh,
    total_portfolio_value=port_value
    )
if allocmo:
    allocationh = dah.greedy_portfolio()[0]
    non_trading_cashh=dah.greedy_portfolio()[1]
else:
    allocationh = dah.lp_portfolio()[0]
    non_trading_cashh=dah.lp_portfolio()[1]
# Put the stocks and the number of shares from the portfolio in a df
symbol_listh = []
cwh=[]
num_shares_listh = []
l_priceh=[]
tot_cashh=[]
for symbolh, num_sharesh in allocationh.items():
    symbol_listh.append(symbolh)
    cwh.append(round(cleaned_weightsh[symbolh],3))
    num_shares_listh.append(num_sharesh)
    l_priceh.append(round(latest_pricesh[symbolh],2))
    tot_cashh.append(round(num_sharesh*latest_pricesh[symbolh],2))
    
df_buyh=pd.DataFrame()
df_buyh['stock']=symbol_listh
df_buyh['weights']=cwh
df_buyh['shares']=num_shares_listh
df_buyh['price']=l_priceh
df_buyh['value']=tot_cashh

c3.write(df_buyh)
c3.write('Επενδυμένο σε μετοχές {0:.2f}€ ή το {1:.2f}% του χαρτοφυλακίου'.format(df_buyh['value'].sum(),100*df_buyh['value'].sum()/port_value))
c3.write('Εναπομείναντα μετρητά :{0:.2f}€ ή το {1:.2f}% του χαρτοφυλακίου'.format(port_value-df_buyh['value'].sum(),100-100*df_buyh['value'].sum()/port_value))
df_buyh=df_buyh.append({'stock':'CASH','weights': round(1-df_buyh['value'].sum()/port_value,2),'shares':0,'price':0,'value':round(port_value-df_buyh['value'].sum(),2)}, ignore_index=True)
c3.write('Εάν θέλεις να σώσεις το παραπάνω χαρτοφυλάκιο τότε δώσε ένα όνομα και ένα email και μετά πάτησε το κουμπί για να σου αποσταλεί σαν αρχείο.')
filenm3=c3.text_input('Δώσε ένα όνομα στο Χαρτοφυλάκιο', value="Portfolio3",key=1)
receiver_email3=c3.text_input('Ποιό είναι το email στο οποίο θα αποσταλεί το χαρτοφυλάκιο?',value='example@example.com',key=6)
filenm3=filenm3+'.csv'
df_buyh.to_csv(filenm3)
if c3.button('Σώσε αυτό το Χαρτοφυλάκιο τύπου 3',key=3):
    if '@' in parseaddr(receiver_email)[1]:
        send_portfolio_byemail(filenm3,receiver_email3)

#-----Χαρτοφυλάκιο Νο4-------------------------------
st.sidebar.write('Παράμετροι για το Χαρτοφυλάκιο Νο4')
c4.subheader('Χαρτοφυλάκιο Νο4')
c4.write('Το προτεινόμενο χαρτοφυλάκιο με τις μετοχές μέγιστης κεφαλοποιημένης απόδοσης για τις Χ ημέρες έχει τα παρακάτω χαρακτηριστικά')
mc=st.sidebar.slider('Από μετοχές ελάχιστης συσσωρευμένης απόδοσης τουλάχιστον Ν% στις τελευταίες Χ ημέρες.',10,int(max_ret),20,1)
c4.write('Αρχική Αξία Χαρτοφυλακίου : '+str(port_value)+'€')
c4.write('Επιλέγουμε από μετοχές με συσσωρευμένη απόδοση τουλάχιστον '+str(mc)+'%')
mc=float(mc/100+1)
m_c=[]
for rw in stocks:
    if float(m_cum_ret[rw])>=mc:
        m_c.append(rw)
hi_cum_universe=m_c

# Create a df with just the stocks from the universe
df_cum = select_columns(df_t, hi_cum_universe)
muc = expected_returns.mean_historical_return(df_cum)
if riskmo:
    Sc = CovarianceShrinkage(df_cum).ledoit_wolf()
else:
    Sc = risk_models.sample_cov(df_cum)
# Optimise the portfolio 
efc = EfficientFrontier(muc, Sc, gamma=2) # Use regularization (gamma=1)
if weightsmo:
    weightsc = efc.max_sharpe()
else:
    weightsc = efc.min_volatility()
cleaned_weightsc = efc.clean_weights(cutoff=cutoff,rounding=3)
efc.portfolio_performance()

c4.write('Sharpe Ratio: '+str(round(efc.portfolio_performance()[2],2)))
c4.write('Απόδοση Χαρτοφυλακίο: '+str(round(efc.portfolio_performance()[0]*100,2))+'%')
c4.write('Μεταβλητότητα Χαρτοφυλακίου: '+str(round(efc.portfolio_performance()[1]*100,2))+'%')
# Allocate
latest_pricesc = get_latest_prices(df_cum)
dac =DiscreteAllocation(
    cleaned_weightsc,
    latest_pricesc,
    total_portfolio_value=port_value
    )
if allocmo:
    allocationc = dac.greedy_portfolio()[0]
    non_trading_cashc=dac.greedy_portfolio()[1]
else:
    allocationc = dac.lp_portfolio()[0]
    non_trading_cashc=dac.lp_portfolio()[1]
# Put the stocks and the number of shares from the portfolio in a df
symbol_listc = []
cwc =[]
num_shares_listc = []
l_pricec=[]
tot_cashc=[]
for symbolc, num_sharesc in allocationc.items():
    symbol_listc.append(symbolc)
    cwc.append(round(cleaned_weightsc[symbolc],3))
    num_shares_listc.append(num_sharesc)
    l_pricec.append(round(latest_pricesc[symbolc],2))
    tot_cashc.append(round(num_sharesc*latest_pricesc[symbolc],2))
    
df_buyc=pd.DataFrame()
df_buyc['stock']=symbol_listc
df_buyc['weights']=cwc
df_buyc['shares']=num_shares_listc
df_buyc['price']=l_pricec
df_buyc['value']=tot_cashc

c4.write(df_buyc)
c4.write('Επενδυμένο σε μετοχές {0:.2f}€ ή το {1:.2f}% του χαρτοφυλακίου'.format(df_buyc['value'].sum(),100*df_buyc['value'].sum()/port_value))
c4.write('Εναπομείναντα μετρητά :{0:.2f}€ ή το {1:.2f}% του χαρτοφυλακίου'.format(port_value-df_buyc['value'].sum(),100-100*df_buyc['value'].sum()/port_value))
df_buyc=df_buyc.append({'stock':'CASH','weights': round(1-df_buyc['value'].sum()/port_value,2),'shares':0,'price':0,'value':round(port_value-df_buyc['value'].sum(),2)}, ignore_index=True)
c4.write('Εάν θέλεις να σώσεις το παραπάνω χαρτοφυλάκιο τότε δώσε ένα όνομα και ένα email και μετά πάτησε το κουμπί για να σου αποσταλεί σαν αρχείο.')
filenm4=c4.text_input('Δώσε ένα όνομα στο Χαρτοφυλάκιο', value="Portfolio4",key=1)
receiver_email4=c4.text_input('Ποιό είναι το email στο οποίο θα αποσταλεί το χαρτοφυλάκιο?',value='example@example.com',key=8)
filenm4=filenm4+'.csv'
df_buyc.to_csv(filenm4)
if c4.button('Σώσε αυτό το Χαρτοφυλάκιο τύπου 4',key=4):
    if '@' in parseaddr(receiver_email)[1]:
        send_portfolio_byemail(filenm4,receiver_email4)

#-----------------------------------------------
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
        
      
  
