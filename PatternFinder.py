

import os
from sklearn.linear_model import LinearRegression
import numpy as np
import multiprocessing as mp
# create these folders if they does not exist
if not os.path.isdir("results"):
    os.mkdir("results")
if not os.path.isdir("logs"):
    os.mkdir("logs")
if not os.path.isdir("data"):
    os.mkdir("data")

def auto_multi(data, tasks_ls, pro_num):
    '''
    Args:
        data (dataframe): csv data after `process_data` function.

    Returns:
        res_ls (dict):
            keys: 'evening', 'morning', ...
            values: [0, 0, 1, 0, 0, ...], [], ...
    '''
    tasks_num = len(tasks_ls.copy())
    
    # Create a queue for all tasks
    q = mp.Queue()

    # Maximum of ps_ls & tmp_ls depends on `pro_num`
    # Maximum of res_ls depends on number of `tasks_num`
    ps_ls = []
    res_ls, tmp_ls = [], []
    while True:
        # termination condition
        if len(res_ls) >= tasks_num:
            break

        if (len(ps_ls) >= pro_num):
            
            for p in ps_ls:
                p.start()

            while True:
                # termination condition
                if len(tmp_ls) >= pro_num:
                    res_ls.extend(tmp_ls)
                    tmp_ls.clear()
                    break

                if q.qsize():
                    tmp_res = q.get()
                    tmp_ls.append(tmp_res)
                    
            for p in ps_ls:
                p.join()

            ps_ls.clear()

        if tasks_ls:
            quotient = len(tasks_ls) // pro_num
            remainder = len(tasks_ls) % pro_num
            if quotient:
                for _ in range(pro_num):
                    task = tasks_ls.pop()
                    ps_ls.append(mp.Process(target=task, args=(data, q, True,)))
            else:
                for _ in range(remainder):
                    task = tasks_ls.pop()
                    ps_ls.append(mp.Process(target=task, args=(data, q, True,)))
                pro_num = remainder
    
    return res_ls

import os
import time
date_now = time.strftime("%Y-%m-%d")

from sklearn.linear_model import LinearRegression
import numpy as np
import multiprocessing as mp
# create these folders if they does not exist
if not os.path.isdir("results"):
    os.mkdir("results")
if not os.path.isdir("logs"):
    os.mkdir("logs")
if not os.path.isdir("data"):
    os.mkdir("data")


if os.path.exists(os.path.join("results",f"{date_now}_" + "ResultsTable-Sum.csv")):
    os.remove(os.path.join("results", f"{date_now}_" + "ResultsTable-Sum.csv"))

INCLUDE_TODAY=0  ## INCLUDE TODAY =0 PRV DAY = 1 PRV 2 DAY=2

tickers=['GRASIM.NS',  'COALINDIA.NS',  'ONGC.NS', 'LT.NS',  'IOC.NS',  'ADANIPORTS.NS',  'HDFC.NS',  
         'BPCL.NS',  'BAJAJFINSV.NS',  'BAJFINANCE.NS',  'TATACONSUM.NS',  'HEROMOTOCO.NS',  'TATAMOTORS.NS',  
         'TECHM.NS',  'M&M.NS',  'NTPC.NS',  'WIPRO.NS',  'DIVISLAB.NS',  'HDFCLIFE.NS',  'JSWSTEEL.NS',
          'INDUSINDBK.NS',  'SBILIFE.NS',  'MARUTI.NS',  'BHARTIARTL.NS',  'ULTRACEMCO.NS',  'BRITANNIA.NS', 
         'SUNPHARMA.NS',  'UPL.NS',  'POWERGRID.NS',  'HCLTECH.NS',  'KOTAKBANK.NS',  'TCS.NS',  'ITC.NS', 
         'CIPLA.NS',  'SHREECEM.NS',  'DRREDDY.NS',  'RELIANCE.NS',  'INFY.NS',  'BAJAJ-AUTO.NS',  
         'ASIANPAINT.NS',  'TATASTEEL.NS',  'EICHERMOT.NS',  'AXISBANK.NS', 
         'SBIN.NS',  'HDFCBANK.NS',  'HINDUNILVR.NS',  'HINDALCO.NS',  'ICICIBANK.NS','TITAN.NS','NESTLEIND.NS',
         'ASHOKLEY.NS']
tickers=['L&TFH.NS',
'TATACONSUM.NS',
'HDFC.NS',
'LTTS.NS',
'TECHM.NS',
'GLENMARK.NS',
'LUPIN.NS',
'APOLLOHOSP.NS',
'EXIDEIND.NS',
'TATACHEM.NS',
'GMRINFRA.NS',
'JUBLFOOD.NS',
'PETRONET.NS',
'TVSMOTOR.NS',
'HEROMOTOCO.NS',
'HINDPETRO.NS',
'PVR.NS',
'BAJFINANCE.NS',
'DRREDDY.NS',
'CIPLA.NS']

tickers=['IDEA.NS',
'UBL.NS',
'MINDTREE.NS',
'JUBLFOOD.NS',
'LALPATHLAB.NS',
'METROPOLIS.NS',
'BHEL.NS',
'CANBK.NS',
'EICHERMOT.NS',
'COROMANDEL.NS',
'ALKEM.NS',
'ULTRACEMCO.NS',
'CUB.NS',
'CADILAHC.NS',
'ICICIGI.NS',
'COFORGE.NS',
'BAJFINANCE.NS',
'ACC.NS',
'AMBUJACEM.NS',
'IRCTC.NS',
'ADANIPORTS.NS',
'BAJAJFINSV.NS',
'INDUSTOWER.NS',
'BATAINDIA.NS',
'RAMCOCEM.NS',
'TRENT.NS',
'TORNTPHARM.NS',
'GRASIM.NS',
'ADANIENT.NS',
'CONCOR.NS',
'IDFCFIRSTB.NS',
'SRTRANSFIN.NS',
'TATACONSUM.NS',
'CUMMINSIND.NS',
'PFIZER.NS',
'NESTLEIND.NS',
'BIOCON.NS',
'BANKBARODA.NS',
'TORNTPOWER.NS',
'ASTRAL.NS',
'HAVELLS.NS',
'MRF.NS',
'PAGEIND.NS',
'NAVINFLUOR.NS',
'BAJAJ-AUTO.NS',
'BRITANNIA.NS',
'ICICIPRULI.NS',
'HDFCAMC.NS',
'MARICO.NS',
'DLF.NS',
'NAUKRI.NS',
'SHREECEM.NS',
'AARTIIND.NS',
'IOC.NS',
'LTI.NS',
'CIPLA.NS',
'PIDILITIND.NS',
'RELIANCE.NS',
'MUTHOOTFIN.NS',
'TITAN.NS',
'RECLTD.NS',
'HDFCLIFE.NS',
'TCS.NS',
'UPL.NS',
'DEEPAKNTR.NS',
'DABUR.NS',
'MPHASIS.NS',
'HINDUNILVR.NS',
'M&M.NS',
'HINDPETRO.NS',
'L&TFH.NS',
'NAM-INDIA.NS',
'DIVISLAB.NS',
'AMARAJABAT.NS',
'PEL.NS',
'PETRONET.NS',
'FEDERALBNK.NS',
'MGL.NS',
'GODREJCP.NS',
'LTTS.NS',
'GLENMARK.NS',
'TECHM.NS',
'VOLTAS.NS',
'HDFCBANK.NS',
'BOSCHLTD.NS',
'ITC.NS',
'BERGEPAINT.NS',
'LUPIN.NS',
'SUNPHARMA.NS',
'EXIDEIND.NS',
'TATASTEEL.NS',
'ABFRL.NS',
'PFC.NS',
'^NSEI',
'^NSEBANK',
'DRREDDY.NS',
'M&MFIN.NS',
'AXISBANK.NS',
'BPCL.NS',
'NMDC.NS',
'LT.NS',
'PVR.NS',
'SBIN.NS',
'INDHOTEL.NS',
'ASIANPAINT.NS',
'INFY.NS',
'INDIGO.NS',
'TATAPOWER.NS',
'CHOLAFIN.NS',
'GMRINFRA.NS',
'BHARTIARTL.NS',
'STAR.NS',
'ONGC.NS',
'COALINDIA.NS',
'MARUTI.NS',
'TATACHEM.NS',
'HEROMOTOCO.NS',
'RBLBANK.NS',
'PNB.NS',
'GRANULES.NS',
'NTPC.NS',
'HCLTECH.NS',
'LICHSGFIN.NS',
'AUBANK.NS',
'PIIND.NS',
'APOLLOTYRE.NS',
'MOTHERSUMI.NS',
'WIPRO.NS',
'JSWSTEEL.NS',
'TVSMOTOR.NS',
'GAIL.NS',
'GUJGASLTD.NS',
'BHARATFORG.NS',
'APLLTD.NS',
'MFSL.NS',
'COLPAL.NS',
'ASHOKLEY.NS',
'HDFC.NS',
'GODREJPROP.NS',
'INDUSINDBK.NS',
'BANDHANBNK.NS',
'SIEMENS.NS',
'SUNTV.NS',
'BALKRISIND.NS',
'BEL.NS',
'ZEEL.NS',
'SAIL.NS',
'IGL.NS',
'AUROPHARMA.NS',
'POWERGRID.NS',
'TATAMOTORS.NS',
'SRF.NS',
'ESCORTS.NS',
'JINDALSTEL.NS',
'SBILIFE.NS',
'ICICIBANK.NS',
'MANAPPURAM.NS',
'KOTAKBANK.NS',
'HINDALCO.NS',
'VEDL.NS',
'NATIONALUM.NS',
'APOLLOHOSP.NS',
'IBULHSGFIN.NS'
]

#n50 only

tickers=['BAJAJFINSV.NS',
'HINDALCO.NS',
'ADANIPORTS.NS',
'TATASTEEL.NS',
'BAJFINANCE.NS',
'TECHM.NS',
'IOC.NS',
'HDFCBANK.NS',
'SBILIFE.NS',
'CIPLA.NS',
'SBIN.NS',
'COALINDIA.NS',
'LT.NS',
'UPL.NS',
'INDUSINDBK.NS',
'SUNPHARMA.NS',
'ICICIBANK.NS',
'DRREDDY.NS',
'GRASIM.NS',
'TATAMOTORS.NS',
'M&M.NS',
'ONGC.NS',
'AXISBANK.NS',
'BAJAJ-AUTO.NS',
'ULTRACEMCO.NS',
'JSWSTEEL.NS',
'BPCL.NS',
'SHREECEM.NS',
'RELIANCE.NS',
'TATACONSUM.NS',
'WIPRO.NS',
'NTPC.NS',
'POWERGRID.NS',
'HDFCLIFE.NS',
'TITAN.NS',
'EICHERMOT.NS',
'HEROMOTOCO.NS',
'ITC.NS',
'MARUTI.NS',
'DIVISLAB.NS',
'BHARTIARTL.NS',
'TCS.NS',
'HINDUNILVR.NS',
'KOTAKBANK.NS',
'HCLTECH.NS',
'INFY.NS',
'HDFC.NS',
'ASIANPAINT.NS',
'BRITANNIA.NS',
'NESTLEIND.NS'
]

#tickers=['BHEL.NS','BIOCON.NS','NAVINFLUOR.NS','ICICIPRULI.NS','RECLTD.NS','INFY.NS','POWERGRID.NS']
#tickers=['BHEL.NS']
def ImportData(ticker,NoOfDays):
# pip install yahoo_fin
  import pandas as pd
  from yahoo_fin import stock_info as si
  from datetime import datetime, timedelta
  d = datetime.today() - timedelta(days=NoOfDays)
  # load it from yahoo_fin library

  if INCLUDE_TODAY == 0 :
    dToday=datetime.today()
  else :
    dToday=datetime.today()-timedelta(days=INCLUDE_TODAY)

  # load it from yahoo
  df = si.get_data(ticker,d,dToday)
  df = df[df.close.notnull()]
  df = df[df.close != 0]
  df.columns=['Open','High','Low','Close','adjclose','Volume','ticker']
  if "Date" not in df.columns:
    df["Date"] = df.index
  df['Date'] = pd.to_datetime(df.Date,format='%Y-%m-%d')
  print("\n Dowloading Data .... : ",ticker)
  #df.to_csv(os.path.join("results", ticker  + "_WorkingData.csv"))
  return df


  


def auto_multi(data, tasks_ls, pro_num):
    '''
    Args:
        data (dataframe): csv data after `process_data` function.

    Returns:
        res_ls (dict):
            keys: 'evening', 'morning', ...
            values: [0, 0, 1, 0, 0, ...], [], ...
    '''
    tasks_num = len(tasks_ls.copy())
    
    # Create a queue for all tasks
    q = mp.Queue()

    # Maximum of ps_ls & tmp_ls depends on `pro_num`
    # Maximum of res_ls depends on number of `tasks_num`
    ps_ls = []
    res_ls, tmp_ls = [], []
    while True:
        # termination condition
        if len(res_ls) >= tasks_num:
            break

        if (len(ps_ls) >= pro_num):
            
            for p in ps_ls:
                p.start()

            while True:
                # termination condition
                if len(tmp_ls) >= pro_num:
                    res_ls.extend(tmp_ls)
                    tmp_ls.clear()
                    break

                if q.qsize():
                    tmp_res = q.get()
                    tmp_ls.append(tmp_res)
                    
            for p in ps_ls:
                p.join()

            ps_ls.clear()

        if tasks_ls:
            quotient = len(tasks_ls) // pro_num
            remainder = len(tasks_ls) % pro_num
            if quotient:
                for _ in range(pro_num):
                    task = tasks_ls.pop()
                    ps_ls.append(mp.Process(target=task, args=(data, q, True,)))
            else:
                for _ in range(remainder):
                    task = tasks_ls.pop()
                    ps_ls.append(mp.Process(target=task, args=(data, q, True,)))
                pro_num = remainder
    
    return res_ls

from tqdm import trange, tqdm
from scipy import stats
import pandas as pd
import numpy as np
import time

#from utils import util_multi as mul
#from utils import util_process as pro


def rename(data):
    rename_dc = {'Gmt time': 'timestamp'}
    data.rename(columns=rename_dc, inplace=True)
    data.columns = [c.lower() for c in data.columns]
    return data


def process_data(data, slope=True):
    '''Including calculation of CLUR, Quartiles, and cus trend
    Args:
        data (dataframe): csv data from assets. With column names open, high, low, close.

    Returns:
        dataframe.
    '''
    if slope:
        # process slpoe
        data['diff'] = data['close'] - data['open']
        data = data.query('diff != 0').reset_index(drop=True)
        data['direction'] = np.sign(data['diff'])
        data['ushadow_width'] = 0
        data['lshadow_width'] = 0

        for idx in trange(len(data)):
            if data.loc[idx, 'direction'] == 1:
                data.loc[idx, 'ushadow_width'] = data.loc[idx, 'high'] - data.loc[idx, 'close']
                data.loc[idx, 'lshadow_width'] = data.loc[idx, 'open'] - data.loc[idx, 'low']
            else:
                data.loc[idx, 'ushadow_width'] = data.loc[idx, 'high'] - data.loc[idx, 'open']
                data.loc[idx, 'lshadow_width'] = data.loc[idx, 'close'] - data.loc[idx, 'low']

            if idx <= 50:
                data.loc[idx, 'body_per'] = stats.percentileofscore(abs(data['diff']), abs(data.loc[idx,'diff']), 'rank')
                data.loc[idx, 'upper_per'] = stats.percentileofscore(data['ushadow_width'], data.loc[idx,'ushadow_width'], 'rank')
                data.loc[idx, 'lower_per'] = stats.percentileofscore(data['lshadow_width'], data.loc[idx,'lshadow_width'], 'rank')
            else:
                data.loc[idx, 'body_per'] = stats.percentileofscore(abs(data.loc[idx-50:idx, 'diff']),abs(data.loc[idx, 'diff']), 'rank')
                data.loc[idx, 'upper_per'] = stats.percentileofscore(data.loc[idx-50:idx, 'ushadow_width'], data.loc[idx, 'ushadow_width'], 'rank')
                data.loc[idx, 'lower_per'] = stats.percentileofscore(data.loc[idx-50:idx, 'lshadow_width'], data.loc[idx, 'lshadow_width'], 'rank')

        data['slope'] = data['close'].rolling(7).apply(get_slope, raw=False)
        data.dropna(inplace=True)
    else:
        # process trend
        data['trend'] = data['slope'].rolling(1).apply(get_trend, raw=False)
        data['previous_trend'] = data['trend'].shift(1).fillna(0)
    return data


def detect_evening_star(data, q=None, multi=False, short_per=35, long_per=65):
    '''Detect evening star pattern    
    Args:
        short_per (int): percentile for determination.
        long_per (int): percentile for determination.

    Returns:
        dataframe.
    '''
    #print('[ Info ] : detecting evening star')
    temp = data[(data['previous_trend'] == 1) & (data['direction'] == 1)].index
    data['evening'] = 0
    try:
        for idx in tqdm(temp):
            cond1 = (data.loc[idx, 'body_per'] >= long_per)
            cond2 = (data.loc[idx+1, 'body_per'] <= short_per)
            cond3 = (data.loc[idx+2, 'direction'] == -1)
            cond4 = (data.loc[idx+1, 'close'] + data.loc[idx+1, 'open'])/2 >= data.loc[idx, 'close']
            cond5 = data.loc[idx+2, 'close'] <= ((data.loc[idx, 'open'] + data.loc[idx, 'close'])/2)
            # cond6 = (data.loc[idx+2, 'body_per'] >= long_per)
            cond7 = (data.loc[idx+2, 'open'] <= (data.loc[idx+1, 'open'] + data.loc[idx+1, 'close'])/2)
            if cond1 & cond2 & cond3 & cond4 & cond5 & cond7:
                data.loc[idx+2, 'evening'] = 1
    except:
        pass

    if multi:
        q.put({'evening': np.array(data['evening'])})
    else:
        return data


def detect_morning_star(data, q=None, multi=False, short_per=35, long_per=65):
    '''Detect morning star pattern    
    Args:
        short_per (int): percentile for determination.
        long_per (int): percentile for determination.

    Returns:
        dataframe.
    '''
    #print('[ Info ] : detecting morning star')
    temp = data[(data['previous_trend'] == -1) & (data['direction'] == -1)].index
    data['morning'] = 0
    try:
        for idx in tqdm(temp):
            cond1 = (data.loc[idx, 'body_per'] >= long_per)
            cond2 = (data.loc[idx+1, 'body_per'] <= short_per)
            cond3 = (data.loc[idx+2, 'direction'] == 1)
            # cond4 = max(data.loc[idx+1, 'close'], data.loc[idx+1, 'open']) <= data.loc[idx, 'close']
            cond4 = (data.loc[idx+1, 'close'] + data.loc[idx+1, 'open'])/2 <= data.loc[idx, 'close']
            cond5 = data.loc[idx+2, 'close'] >= ((data.loc[idx, 'open'] + data.loc[idx, 'close'])/2)
            # cond6 = (data.loc[idx+2, 'body_per'] >= long_per)
            cond7 = (data.loc[idx+2, 'open'] >= (data.loc[idx+1, 'open'] + data.loc[idx+1, 'close'])/2)
            if cond1 & cond2 & cond3 & cond4 & cond5 & cond7:
                data.loc[idx+2, 'morning'] = 1
    except:
        pass

    if multi:
        q.put({'morning': np.array(data['morning'])})
    else:
        return data


def detect_shooting_star(data, q=None, multi=False, short_per=35, long_per=65):
    '''Detect shooting star pattern    
    Args:
        short_per (int): percentile for determination.
        long_per (int): percentile for determination.

    Returns:
        dataframe.
    '''
    #print('[ Info ] : detecting shooting star')
    data['shooting_star'] = 0
    temp = data[(data['previous_trend'] == 1) & (data['direction'] == 1)].index
    try:
        for idx in tqdm(temp):
            cond1 = (data.loc[idx, 'body_per'] >= long_per)
            cond2 = (data.loc[idx, 'direction'] == 1)
            cond3 = (data.loc[idx+1, 'ushadow_width'] > 2 * abs(data.loc[idx+1, 'diff']))
            cond4 = (min(data.loc[idx+1, 'open'], data.loc[idx+1, 'close']) > ((data.loc[idx, 'close'] + data.loc[idx, 'open']) / 2))
            cond5 = (data.loc[idx+1, 'lower_per'] <= short_per - 10)  # 25
            cond6 = (data.loc[idx+1, 'upper_per'] >= long_per)
            if cond1 & cond2 & cond3 & cond4 & cond5 & cond6:
                data.loc[idx+1, 'shooting_star'] = 1
    except:
        pass

    if multi:
        q.put({'shooting_star': np.array(data['shooting_star'])})
    else:
        return data


def detect_hanging_man(data, q=None, multi=False, short_per=35, long_per=65):
    '''Detect hanging man pattern    
    Args:
        short_per (int): percentile for determination.
        long_per (int): percentile for determination.

    Returns:
        dataframe.
    '''
    #print('[ Info ] : detecting hanging man')
    data['hanging_man'] = 0
    temp = data[(data['previous_trend'] == 1) & (data['direction'] == 1)].index
    try:
        for idx in tqdm(temp):
            cond1 = (data.loc[idx, 'lshadow_width'] > 2 * abs(data.loc[idx, 'diff']))
            cond2 = (data.loc[idx, 'body_per'] <= short_per)
            cond3 = (data.loc[idx, 'upper_per'] <= (short_per - 10))
            cond4 = (data.loc[idx, 'lower_per'] >= long_per)
            if cond1 & cond2 & cond3 & cond4:
                data.loc[idx, 'hanging_man'] = 1
    except:
        pass

    if multi:
        q.put({'hanging_man': np.array(data['hanging_man'])})
    else:
        return data


def detect_bullish_engulfing(data, q=None, multi=False, short_per=35, long_per=65):
    '''Detect bullish engulfing pattern
    Args:
        short_per (int): percentile for determination.
        long_per (int): percentile for determination.
    
    Returns:
        dataframe.
    '''
    #print('[ Info ] : detecting bullish engulfing')
    data['bullish_engulfing'] = 0
    temp = data[(data['previous_trend'] == -1) & (data['direction'] == -1)].index
    try:
        for idx in tqdm(temp):
            cond1 = (data.loc[idx, 'direction'] == -1)
            cond2 = (data.loc[idx, 'body_per'] >= long_per)
            cond3 = (data.loc[idx+1, 'direction'] == 1)
            cond4 = (data.loc[idx+1, 'close'] > data.loc[idx, 'open'])
            cond5 = (data.loc[idx+1, 'open'] < data.loc[idx, 'close'])
            if cond1 & cond2 & cond3 & cond4 & cond5:
                data.loc[idx+1, 'bullish_engulfing'] = 1
    except:
        pass

    if multi:
        q.put({'bullish_engulfing': np.array(data['bullish_engulfing'])})
    else:
        return data


def detect_bearish_engulfing(data, q=None, multi=False, short_per=35, long_per=65):
    '''Detect bearish engulfing pattern
    Args:
        short_per (int): percentile for determination.
        long_per (int): percentile for determination.

    Returns:
        dataframe.
    '''
    #print('[ Info ] : detecting bearish engulfing')
    data['bearish_engulfing'] = 0
    temp = data[(data['previous_trend'] == 1) & (data['direction'] == 1)].index
    try:
        for idx in tqdm(temp):
            cond1 = (data.loc[idx, 'direction'] == 1)
            cond2 = (data.loc[idx, 'body_per'] >= long_per)
            cond3 = (data.loc[idx+1, 'direction'] == -1)
            cond4 = (data.loc[idx+1, 'close'] < data.loc[idx, 'open'])
            cond5 = (data.loc[idx+1, 'open'] > data.loc[idx, 'close'])
            if cond1 & cond2 & cond3 & cond4 & cond5:
                data.loc[idx+1, 'bearish_engulfing'] = 1
    except:
        pass

    if multi:
        q.put({'bearish_engulfing': np.array(data['bearish_engulfing'])})
    else:
        return data


def detect_hammer(data, q=None, multi=False, short_per=35, long_per=65):
    '''Detect hammer pattern    
    Args:
        short_per (int): percentile for determination.
        long_per (int): percentile for determination.

    Returns:
        dataframe.
    '''
   # print('[ Info ] : detecting hammer')
    data['hammer'] = 0
    temp = data[(data['previous_trend'] == -1) & (data['direction'] == -1)].index
    try:
        for idx in tqdm(temp):
            cond1 = (data.loc[idx, 'lshadow_width'] > 2 * abs(data.loc[idx, 'diff']))
            cond2 = (data.loc[idx, 'body_per'] <= short_per)
            cond3 = (data.loc[idx, 'upper_per'] <= (short_per - 15))
            cond4 = (data.loc[idx, 'lower_per'] >= long_per)
            if cond1 & cond2 & cond3 & cond4:
                data.loc[idx, 'hammer'] = 1
    except:
        pass

    if multi:
        q.put({'hammer': np.array(data['hammer'])})
    else:
        return data    


def detect_inverted_hammer(data, q=None, multi=False, short_per=35, long_per=65):
    '''Detect inverted hammer pattern    
    Args:
        short_per (int): percentile for determination.
        long_per (int): percentile for determination.

    Returns:
        dataframe.
    '''
    #print('[ Info ] : detecting inverted hammer')
    data['inverted_hammer'] = 0
    temp = data[(data['previous_trend'] == -1) & (data['direction'] == -1)].index
    try:
        for idx in tqdm(temp):
            cond1 = (data.loc[idx, 'direction'] == -1)
            cond2 = (data.loc[idx, 'body_per'] >= long_per)
            cond3 = (data.loc[idx+1, 'ushadow_width'] > 2 * abs(data.loc[idx+1, 'diff']))
            cond4 = (max(data.loc[idx+1, 'open'], data.loc[idx+1, 'close']) < ((data.loc[idx, 'close'] + data.loc[idx, 'open']) / 2))
            cond5 = (data.loc[idx+1, 'lower_per'] <= short_per)
            cond6 = (data.loc[idx+1, 'upper_per'] >= long_per)
            if cond1 & cond2 & cond3 & cond4 & cond5 & cond6:
                data.loc[idx+1, 'inverted_hammer'] = 1
    except:
        pass

    if multi:
        q.put({'inverted_hammer': np.array(data['inverted_hammer'])})
    else:
        return data    


def detect_bullish_harami(data, q=None, multi=False, short_per=35, long_per=65):
    '''Detect inverted bullish harami pattern    
    Args:
        short_per (int): percentile for determination.
        long_per (int): percentile for determination.
        
    Returns:
        dataframe.
    '''
   # print('[ Info ] : detecting bullish harami')
    data['bullish_harami'] = 0
    temp = data[(data['previous_trend'] == -1) & (data['direction'] == -1)].index
    try:
        for idx in tqdm(temp):
            cond1 = (data.loc[idx, 'direction'] == -1)
            cond2 = (data.loc[idx, 'body_per'] >= long_per)
            cond3 = (data.loc[idx+1, 'direction'] == 1)
            cond4 = (data.loc[idx+1, 'close'] >= ((data.loc[idx, 'open'] + data.loc[idx, 'close'])/2))
            cond5 = (data.loc[idx+1, 'close'] < data.loc[idx, 'open'])
            cond6 = (data.loc[idx+1, 'open'] > data.loc[idx, 'close'])
            cond7 = (data.loc[idx+1, 'open'] <= ((data.loc[idx, 'open'] + data.loc[idx, 'close'])/2))
            cond8 = (data.loc[idx+1, 'body_per'] >= long_per)
            if cond1 & cond2 & cond3 & cond4 & cond5 & cond6 & cond7 & cond8:
                data.loc[idx+1, 'bullish_harami'] = 1
    except:
        pass

    if multi:
        q.put({'bullish_harami': np.array(data['bullish_harami'])})
    else:
        return data    


def detect_bearish_harami(data, q=None, multi=False, short_per=35, long_per=65):
    '''Detect inverted bearish harami pattern    
    Args:
        short_per (int): percentile for determination.
        long_per (int): percentile for determination.

    Returns:
        dataframe.
    '''
    #print('[ Info ] : detecting bearish harami')
    data['bearish_harami'] = 0
    temp = data[(data['previous_trend'] == 1) & (data['direction'] == 1)].index
    try:
        for idx in tqdm(temp):
            cond1 = (data.loc[idx, 'direction'] == 1)
            cond2 = (data.loc[idx, 'body_per'] >= long_per)
            cond3 = (data.loc[idx+1, 'direction'] == -1)
            cond4 = (data.loc[idx+1, 'close'] <= ((data.loc[idx, 'open'] + data.loc[idx, 'close'])/2))
            cond5 = (data.loc[idx+1, 'close'] > data.loc[idx, 'open'])
            cond6 = (data.loc[idx+1, 'open'] < data.loc[idx, 'close'])
            cond7 = (data.loc[idx+1, 'open'] >= ((data.loc[idx, 'open'] + data.loc[idx, 'close'])/2))
            cond8 = (data.loc[idx+1, 'body_per'] >= long_per)
            if cond1 & cond2 & cond3 & cond4 & cond5 & cond6 & cond7 & cond8:
                data.loc[idx+1, 'bearish_harami'] = 1
    except:
        pass

    if multi:
        q.put({'bearish_harami': np.array(data['bearish_harami'])})
    else:
        return data       


def detect_all(data, tasks_ls=None, multi=False, pro_num=2):
    '''
    Args:
        data (dataframe): csv data after `process_data` function.
        multi (bool): use multiprocessing or not.
        pro_num (int): how many processes to be used.

    Returns:
        data (dataframe): dataframe with detections.
    '''
    if multi:
        res_ls = auto_multi(data, tasks_ls, pro_num)
        #print('[ Info ] join finished !')

        dc = {}
        for res in res_ls:
            for key, value in res.items():
                dc[key] = value
        df = pd.DataFrame(dc)
        data = pd.concat([data, df], axis=1)
    else:
        data = detect_evening_star(data)
        data = detect_morning_star(data)
        data = detect_shooting_star(data)
        data = detect_hanging_man(data)
        data = detect_bullish_engulfing(data)
        data = detect_bearish_engulfing(data)
        data = detect_hammer(data)
        data = detect_inverted_hammer(data)
        data = detect_bullish_harami(data)
        data = detect_bearish_harami(data)
    return data


def detection_result(data):
    '''Print numbers of detection    
    Args:
        data (dataframe): csv data after `process_data` function.

    Returns:
        data (dataframe): dataframe with detections.
    '''
    # print('\n[ Info ] : number of evening star is %s' % np.sum(data['evening']))
    # print('[ Info ] : number of morning star is %s' % np.sum(data['morning']))
    # print('[ Info ] : number of shooting star is %s' % np.sum(data['shooting_star']))
    # print('[ Info ] : number of hanging man is %s' % np.sum(data['hanging_man']))
    # print('[ Info ] : number of bullish engulfing is %s' % np.sum(data['bullish_engulfing']))
    # print('[ Info ] : number of bearish engulfing is %s' % np.sum(data['bearish_engulfing']))
    # print('[ Info ] : number of hammer is %s' % np.sum(data['hammer']))
    # print('[ Info ] : number of inverted hammer is %s' % np.sum(data['inverted_hammer']))
    # print('[ Info ] : number of bullish harami is %s' % np.sum(data['bullish_harami']))
    # print('[ Info ] : number of bearish harami is %s' % np.sum(data['bearish_harami']))



def get_slope(series):
    y = series.values.reshape(-1, 1)
    x = np.array(range(1, series.shape[0] + 1)).reshape(-1,1)
    model = LinearRegression()
    model.fit(x, y)
    slope = model.coef_
    return slope




def get_trend(slope):
    '''Need to run `process_data` first with slope only, then calculate by yourself.
    25 percentile: 7.214285714286977e-05
    '''
    slope = np.array(slope)
    thres = 7.214285714286977e-05
    if (slope >= thres):
        return 1
    elif (slope <= -thres):
        return -1
    else:
        return 0


if __name__ == "__main__":
    TASLS_LS = [detect_evening_star, detect_morning_star, detect_shooting_star, 
                detect_hanging_man, detect_bullish_engulfing, detect_bearish_engulfing,
                detect_hammer, detect_inverted_hammer, detect_bullish_harami,
                detect_bearish_harami]
    
    # load raw ohlc data

for x in range(len(tickers)):
   # print(tickers[x])
    ticker = tickers[x]


    data = ImportData(ticker,152)
    data = rename(data)

    # calculate features & slope
    data = process_data(data, slope=True)

    # calculate trend (depend on slopes)
    data = process_data(data, slope=False)

    
    #print(data.tail(1))
    # detect with customized rules
    data = detect_all(data, TASLS_LS, multi=False, pro_num=4)
   
    cols=['evening', 'morning', 'shooting_star', 'hanging_man', 'bullish_engulfing', 'bearish_engulfing', 'hammer', 'inverted_hammer', 'bullish_harami', 'bearish_harami']
    data["sum"] = data[cols].sum(axis=1)
    data['PatternFound']='SEARCHING..'
    #print(data.loc[data['sum'] >0])
    #data.to_csv(os.path.join("results", "ResultsTable-" + ticker  + ".csv"), index=False)
    
    print('\n Today Patterns Found  : ' + str(data['sum'].iloc[-1]) )
    print(' Total Patterns Found  : ' + str(sum(data['sum'])))
    print(data.tail(1))
    print("Count Of Search:",x)
   
    #print(dfnew['sum'].tail(1))
    dfnew=data[['ticker','date','evening', 'morning', 'shooting_star', 'hanging_man', 'bullish_engulfing', 'bearish_engulfing', 'hammer', 'inverted_hammer', 'bullish_harami', 'bearish_harami','sum','PatternFound']]    
    if dfnew['sum'].iloc[-1]>0 : 
       dfnew=dfnew[dfnew["sum"] > 0]

       if dfnew['evening'].iloc[-1] > 0 :
         dfnew['PatternFound']='Evening Star'
      
       if dfnew['morning'].iloc[-1] > 0 :
        dfnew['PatternFound']='Morning Star'

       if dfnew['shooting_star'].iloc[-1] > 0 :
         dfnew['PatternFound']='Shooting Star'
      
       if dfnew['hanging_man'].iloc[-1] > 0 :
        dfnew['PatternFound']='Hanging man'
       
       if dfnew['bullish_engulfing'].iloc[-1] > 0 :
         dfnew['PatternFound']='Bullish engulfing'
      
       if dfnew['bearish_engulfing'].iloc[-1] > 0 :
        dfnew['PatternFound']='Bearish engulfing'

       if dfnew['hammer'].iloc[-1] > 0 :
         dfnew['PatternFound']='Hammer'
      
       if dfnew['inverted_hammer'].iloc[-1] > 0 :
        dfnew['PatternFound']='Inverted_hammer'

       if dfnew['bullish_harami'].iloc[-1] > 0 :
         dfnew['PatternFound']='Bullish harami'
      
       if dfnew['bearish_harami'].iloc[-1] > 0 :
        dfnew['PatternFound']='Bearish harami'
      


       print("Ticker found pattern :",ticker)
       dfnew=dfnew[["ticker", "date","PatternFound"]]
       dfnew=dfnew.tail(1)  
       if os.path.exists(os.path.join("results",f"{date_now}_" + "ResultsTable-Sum.csv")):  
        dfnew.to_csv(os.path.join("results", f"{date_now}_" + "ResultsTable-Sum.csv") ,mode='a', index=False,header=False  )
       else :
        dfnew.to_csv(os.path.join("results", f"{date_now}_" + "ResultsTable-Sum.csv") ,mode='a', index=False)
       #input("Press Enter to continue...")
    # if len(dfnew)>0:
    #     print("Found Pattern")
    #     input("Press Enter to continue...")
    #     print(dfnew)
    #     input("Press Enter to continue...")
if os.path.exists(os.path.join("results",f"{date_now}_" + "ResultsTable-Sum.csv")):
  dfResults = pd.read_csv(os.path.join("results", f"{date_now}_" + "ResultsTable-Sum.csv"))
  
  

  print(dfResults)
  
  