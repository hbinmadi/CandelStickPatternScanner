import numpy as np
import talib
import os
import pandas as pd
from sklearn.linear_model import LinearRegression
from itertools import compress
if not os.path.isdir("results"):
    os.mkdir("results")
if not os.path.isdir("logs"):
    os.mkdir("logs")
if not os.path.isdir("data"):
    os.mkdir("data")
import numpy
import math
import time
date_now = time.strftime("%Y-%m-%d")
    
INCLUDE_TODAY=0 # 0= start from today
Scanner=4
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
  #print(dToday)
  # load it from yahoo
  df = si.get_data(ticker,d,dToday)
  df = df[df.close.notnull()]
  df = df[df.close != 0]
  df.columns=['Open','High','Low','Close','adjclose','Volume','ticker']
  if "Date" not in df.columns:
    df["Date"] = df.index
  df['Date'] = pd.to_datetime(df.Date,format='%Y-%m-%d')
  #print("\n Dowloading Data .... : ",ticker)
  #df.to_csv(os.path.join("results", ticker  + "_WorkingData.csv"))
  return df


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

candle_rankings = {
        "CDL3LINESTRIKE_Bull": 1,
        "CDL3LINESTRIKE_Bear": 2,
        "CDL3BLACKCROWS_Bull": 3,
        "CDL3BLACKCROWS_Bear": 3,
        "CDLEVENINGSTAR_Bull": 4,
        "CDLEVENINGSTAR_Bear": 4,
        "CDLTASUKIGAP_Bull": 5,
        "CDLTASUKIGAP_Bear": 5,
        "CDLINVERTEDHAMMER_Bull": 6,
        "CDLINVERTEDHAMMER_Bear": 6,
        "CDLMATCHINGLOW_Bull": 7,
        "CDLMATCHINGLOW_Bear": 7,
        "CDLABANDONEDBABY_Bull": 8,
        "CDLABANDONEDBABY_Bear": 8,
        "CDLBREAKAWAY_Bull": 10,
        "CDLBREAKAWAY_Bear": 10,
        "CDLMORNINGSTAR_Bull": 12,
        "CDLMORNINGSTAR_Bear": 12,
        "CDLPIERCING_Bull": 13,
        "CDLPIERCING_Bear": 13,
        "CDLSTICKSANDWICH_Bull": 14,
        "CDLSTICKSANDWICH_Bear": 14,
        "CDLTHRUSTING_Bull": 15,
        "CDLTHRUSTING_Bear": 15,
        "CDLINNECK_Bull": 17,
        "CDLINNECK_Bear": 17,
        "CDL3INSIDE_Bull": 20,
        "CDL3INSIDE_Bear": 56,
        "CDLHOMINGPIGEON_Bull": 21,
        "CDLHOMINGPIGEON_Bear": 21,
        "CDLDARKCLOUDCOVER_Bull": 22,
        "CDLDARKCLOUDCOVER_Bear": 22,
        "CDLIDENTICAL3CROWS_Bull": 24,
        "CDLIDENTICAL3CROWS_Bear": 24,
        "CDLMORNINGDOJISTAR_Bull": 25,
        "CDLMORNINGDOJISTAR_Bear": 25,
        "CDLXSIDEGAP3METHODS_Bull": 27,
        "CDLXSIDEGAP3METHODS_Bear": 26,
        "CDLTRISTAR_Bull": 28,
        "CDLTRISTAR_Bear": 76,
        "CDLGAPSIDESIDEWHITE_Bull": 46,
        "CDLGAPSIDESIDEWHITE_Bear": 29,
        "CDLEVENINGDOJISTAR_Bull": 30,
        "CDLEVENINGDOJISTAR_Bear": 30,
        "CDL3WHITESOLDIERS_Bull": 32,
        "CDL3WHITESOLDIERS_Bear": 32,
        "CDLONNECK_Bull": 33,
        "CDLONNECK_Bear": 33,
        "CDL3OUTSIDE_Bull": 34,
        "CDL3OUTSIDE_Bear": 39,
        "CDLRICKSHAWMAN_Bull": 35,
        "CDLRICKSHAWMAN_Bear": 35,
        "CDLSEPARATINGLINES_Bull": 36,
        "CDLSEPARATINGLINES_Bear": 40,
        "CDLLONGLEGGEDDOJI_Bull": 37,
        "CDLLONGLEGGEDDOJI_Bear": 37,
        "CDLHARAMI_Bull": 38,
        "CDLHARAMI_Bear": 72,
        "CDLLADDERBOTTOM_Bull": 41,
        "CDLLADDERBOTTOM_Bear": 41,
        "CDLCLOSINGMARUBOZU_Bull": 70,
        "CDLCLOSINGMARUBOZU_Bear": 43,
        "CDLTAKURI_Bull": 47,
        "CDLTAKURI_Bear": 47,
        "CDLDOJISTAR_Bull": 49,
        "CDLDOJISTAR_Bear": 51,
        "CDLHARAMICROSS_Bull": 50,
        "CDLHARAMICROSS_Bear": 80,
        "CDLADVANCEBLOCK_Bull": 54,
        "CDLADVANCEBLOCK_Bear": 54,
        "CDLSHOOTINGSTAR_Bull": 55,
        "CDLSHOOTINGSTAR_Bear": 55,
        "CDLMARUBOZU_Bull": 71,
        "CDLMARUBOZU_Bear": 57,
        "CDLUNIQUE3RIVER_Bull": 60,
        "CDLUNIQUE3RIVER_Bear": 60,
        "CDL2CROWS_Bull": 61,
        "CDL2CROWS_Bear": 61,
        "CDLBELTHOLD_Bull": 62,
        "CDLBELTHOLD_Bear": 63,
        "CDLHAMMER_Bull": 65,
        "CDLHAMMER_Bear": 65,
        "CDLHIGHWAVE_Bull": 67,
        "CDLHIGHWAVE_Bear": 67,
        "CDLSPINNINGTOP_Bull": 69,
        "CDLSPINNINGTOP_Bear": 73,
        "CDLUPSIDEGAP2CROWS_Bull": 74,
        "CDLUPSIDEGAP2CROWS_Bear": 74,
        "CDLGRAVESTONEDOJI_Bull": 77,
        "CDLGRAVESTONEDOJI_Bear": 77,
        "CDLHIKKAKEMOD_Bull": 82,
        "CDLHIKKAKEMOD_Bear": 81,
        "CDLHIKKAKE_Bull": 85,
        "CDLHIKKAKE_Bear": 83,
        "CDLENGULFING_Bull": 84,
        "CDLENGULFING_Bear": 91,
        "CDLMATHOLD_Bull": 86,
        "CDLMATHOLD_Bear": 86,
        "CDLHANGINGMAN_Bull": 87,
        "CDLHANGINGMAN_Bear": 87,
        "CDLRISEFALL3METHODS_Bull": 94,
        "CDLRISEFALL3METHODS_Bear": 89,
        "CDLKICKING_Bull": 96,
        "CDLKICKING_Bear": 102,
        "CDLDRAGONFLYDOJI_Bull": 98,
        "CDLDRAGONFLYDOJI_Bear": 98,
        "CDLCONCEALBABYSWALL_Bull": 101,
        "CDLCONCEALBABYSWALL_Bear": 101,
        "CDL3STARSINSOUTH_Bull": 103,
        "CDL3STARSINSOUTH_Bear": 103,
        "CDLDOJI_Bull": 104,
        "CDLDOJI_Bear": 104
    }



N50=['BAJAJFINSV.NS',
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


bnf=['FEDERALBNK.NS',
'PNB.NS',
'INDUSINDBK.NS',
'BANDHANBNK.NS',
'HDFCBANK.NS',
'SBIN.NS',
'KOTAKBANK.NS',
'RBLBANK.NS',
'ICICIBANK.NS',
'IDFCFIRSTB.NS',
'AUBANK.NS',
'AXISBANK.NS'
]

ntop5=['RELIANCE.NS','HDFCBANK.NS',
'SBIN.NS',
'KOTAKBANK.NS',
'ICICIBANK.NS','TCS.NS','HINDUNILVR.NS','HDFC.NS','INFY.NS','AXISBANK.NS','LT.NS']

nfo=['IDEA.NS', 'UBL.NS', 'MINDTREE.NS', 'JUBLFOOD.NS', 'LALPATHLAB.NS', 'METROPOLIS.NS', 'BHEL.NS', 'CANBK.NS', 'EICHERMOT.NS', 'COROMANDEL.NS', 'ALKEM.NS', 'ULTRACEMCO.NS', 'CUB.NS', 'CADILAHC.NS', 'ICICIGI.NS', 'COFORGE.NS', 'BAJFINANCE.NS', 'ACC.NS', 'AMBUJACEM.NS', 'IRCTC.NS', 'ADANIPORTS.NS', 'BAJAJFINSV.NS', 'INDUSTOWER.NS', 'BATAINDIA.NS', 'RAMCOCEM.NS', 'TRENT.NS', 'TORNTPHARM.NS', 'GRASIM.NS', 'ADANIENT.NS', 'CONCOR.NS', 'IDFCFIRSTB.NS', 'SRTRANSFIN.NS', 'TATACONSUM.NS', 'CUMMINSIND.NS', 'PFIZER.NS', 'NESTLEIND.NS', 'BIOCON.NS', 'BANKBARODA.NS', 'TORNTPOWER.NS', 'ASTRAL.NS', 'HAVELLS.NS', 'MRF.NS', 'PAGEIND.NS', 'NAVINFLUOR.NS', 'BAJAJ-AUTO.NS', 'BRITANNIA.NS', 'ICICIPRULI.NS', 'HDFCAMC.NS', 'MARICO.NS', 'DLF.NS', 'NAUKRI.NS', 'SHREECEM.NS', 'AARTIIND.NS', 'IOC.NS', 'LTI.NS', 'CIPLA.NS', 'PIDILITIND.NS', 'RELIANCE.NS', 'MUTHOOTFIN.NS', 'TITAN.NS', 'RECLTD.NS', 'HDFCLIFE.NS', 'TCS.NS', 'UPL.NS', 'DEEPAKNTR.NS', 'DABUR.NS', 'MPHASIS.NS', 'HINDUNILVR.NS', 'M&M.NS', 'HINDPETRO.NS', 'L&TFH.NS', 'NAM-INDIA.NS', 'DIVISLAB.NS', 'AMARAJABAT.NS', 'PEL.NS', 'PETRONET.NS', 'FEDERALBNK.NS', 'MGL.NS', 'GODREJCP.NS', 'LTTS.NS', 'GLENMARK.NS', 'TECHM.NS', 'VOLTAS.NS', 'HDFCBANK.NS', 'BOSCHLTD.NS', 'ITC.NS', 'BERGEPAINT.NS', 'LUPIN.NS', 'SUNPHARMA.NS', 'EXIDEIND.NS', 'TATASTEEL.NS', 'ABFRL.NS', 'PFC.NS', '^NSEI', '^NSEBANK', 'DRREDDY.NS', 'M&MFIN.NS', 'AXISBANK.NS', 'BPCL.NS', 'NMDC.NS', 'LT.NS', 'PVR.NS', 'SBIN.NS', 'INDHOTEL.NS', 'ASIANPAINT.NS', 'INFY.NS', 'INDIGO.NS', 'TATAPOWER.NS', 'CHOLAFIN.NS', 'GMRINFRA.NS', 'BHARTIARTL.NS', 'STAR.NS', 'ONGC.NS', 'COALINDIA.NS', 'MARUTI.NS', 'TATACHEM.NS', 'HEROMOTOCO.NS', 'RBLBANK.NS', 'PNB.NS', 'GRANULES.NS', 'NTPC.NS', 'HCLTECH.NS', 'LICHSGFIN.NS', 'AUBANK.NS', 'PIIND.NS', 'APOLLOTYRE.NS', 'MOTHERSUMI.NS', 'WIPRO.NS', 'JSWSTEEL.NS', 'TVSMOTOR.NS', 'GAIL.NS', 'GUJGASLTD.NS', 'BHARATFORG.NS', 'APLLTD.NS', 'MFSL.NS', 'COLPAL.NS', 'ASHOKLEY.NS', 'HDFC.NS', 'GODREJPROP.NS', 'INDUSINDBK.NS', 'BANDHANBNK.NS', 'SIEMENS.NS', 'SUNTV.NS', 'BALKRISIND.NS', 'BEL.NS', 'ZEEL.NS', 'SAIL.NS', 'IGL.NS', 'AUROPHARMA.NS', 'POWERGRID.NS', 'TATAMOTORS.NS', 'SRF.NS', 'ESCORTS.NS', 'JINDALSTEL.NS', 'SBILIFE.NS', 'ICICIBANK.NS', 'MANAPPURAM.NS', 'KOTAKBANK.NS', 'HINDALCO.NS', 'VEDL.NS', 'NATIONALUM.NS', 'APOLLOHOSP.NS', 'IBULHSGFIN.NS' ] 


if (Scanner==1):
    tickers=N50
elif (Scanner==2):
    tickers=bnf
elif (Scanner==4):
    tickers=nfo
else:
    tickers=ntop5


if os.path.exists(os.path.join("results",f"{date_now}_" + "CandelsSum.csv")):
    os.remove(os.path.join("results", f"{date_now}_" + "CandelsSum.csv"))


def recognize_candlestick(df):
    """
    Recognizes candlestick patterns and appends 2 additional columns to df;
    1st - Best Performance candlestick pattern matched by www.thepatternsite.com
    2nd - # of matched patterns
    """

    op = df['Open'].astype(float)
    hi = df['High'].astype(float)
    lo = df['Low'].astype(float)
    cl = df['Close'].astype(float)

    candle_names = talib.get_function_groups()['Pattern Recognition']
    
    # patterns not found in the patternsite.com
   # patterns not found in the patternsite.com
    #exclude_items =["CDLCOUNTERATTACK","CDLLONGLINE","CDLSHORTLINE","CDLSTALLEDPATTERN","CDLKICKINGBYLENGTH"]
    exclude_items = ('CDLCOUNTERATTACK','CDLLONGLINE','CDLSHORTLINE','CDLSTALLEDPATTERN','CDLKICKINGBYLENGTH')    #candle_names = [candle for candle in candle_names if candle not in exclude_items]
    cnames =  set(candle_names)-set(exclude_items)
    candle_names=list(cnames)
    
    # create columns for each candle
    for candle in candle_names:
        # below is same as;
        # df["CDL3LINESTRIKE"] = talib.CDL3LINESTRIKE(op, hi, lo, cl)
        df[candle] = getattr(talib, candle)(op, hi, lo, cl)

    
    df['candlestick_pattern'] = np.nan
    df['candlestick_match_count'] = np.nan
    for index, row in df.iterrows():

        # no pattern found
        if len(row[candle_names]) - sum(row[candle_names] == 0) == 0:
            df.loc[index,'candlestick_pattern'] = "NO_PATTERN"
            df.loc[index, 'candlestick_match_count'] = 0
        # single pattern found
        elif len(row[candle_names]) - sum(row[candle_names] == 0) == 1:
            # bull pattern 100 or 200
            if any(row[candle_names].values > 0):
                pattern = list(compress(row[candle_names].keys(), row[candle_names].values != 0))[0] + '_Bull'
                df.loc[index, 'UpOrDown'] = 1
                df.loc[index, 'candlestick_pattern'] = pattern
                df.loc[index, 'candlestick_match_count'] = 1
            # bear pattern -100 or -200
            else:
                pattern = list(compress(row[candle_names].keys(), row[candle_names].values != 0))[0] + '_Bear'
                df.loc[index, 'UpOrDown'] = 0
                df.loc[index, 'candlestick_pattern'] = pattern
                df.loc[index, 'candlestick_match_count'] = 0
        # multiple patterns matched -- select best performance
        else:
            # filter out pattern names from bool list of values
            patterns = list(compress(row[candle_names].keys(), row[candle_names].values != 0))
            container = []
            for pattern in patterns:
                if row[pattern] > 0:
                    container.append(pattern + '_Bull')
                    df.loc[index, 'UpOrDown'] = 1
                else:
                    container.append(pattern + '_Bear')
                    df.loc[index, 'UpOrDown'] = 0
            rank_list = [candle_rankings[p] for p in container]
            if len(rank_list) == len(container):
                rank_index_best = rank_list.index(min(rank_list))
                df.loc[index, 'candlestick_pattern'] = container[rank_index_best]
                df.loc[index, 'candlestick_match_count'] = len(container)
                y="".join(container[rank_index_best])
                CheckBullOrBear=y[-4:]
                if CheckBullOrBear == 'Bear' :
                    df.loc[index, 'UpOrDown'] = 0
                else:
                    df.loc[index, 'UpOrDown'] = 1
    # clean up candle columns
    cols_to_drop = list(candle_names)
    df['slope'] = df['Close'].rolling(7).apply(get_slope, raw=False)
    df['trend'] = df['slope'].rolling(1).apply(get_trend, raw=False)
    df['previous_trend'] = df['trend'].shift(1).fillna(0)
    df.drop(cols_to_drop, axis = 1, inplace = True)
   
    if (df['previous_trend'].iloc[-1]>0 and math.isnan(df['UpOrDown'].iloc[-1])) :
          df.loc[index, 'UpOrDown'] = 1
    if (df['previous_trend'].iloc[-1]<0 and math.isnan(df['UpOrDown'].iloc[-1])) :
          df.loc[index, 'UpOrDown'] = 0

    return df
for x in range(len(tickers)):
    dd=ImportData(tickers[x],150)
    dfresults=recognize_candlestick(dd)
    #dfresults['Ticker']=tickers[x]
    #dfresults=dfresults[dfresults["candlestick_match_count"]>0 , dfresults['Date']==date_now] 
    if dfresults['candlestick_match_count'].iloc[-1]!=10 : # !=10 include all >0 candelstick found 
        rslt_df=dfresults.tail(1)
        
        if os.path.exists(os.path.join("results",f"{date_now}_" + "CandelsSum.csv")):  
            rslt_df.to_csv(os.path.join("results", f"{date_now}_" + "CandelsSum.csv") ,mode='a',header=False  )
            
        else :
            rslt_df.to_csv(os.path.join("results", f"{date_now}_" + "CandelsSum.csv") ,mode='a')
            
     
        #print(rslt_df) #.tail(1))
 
    


if os.path.exists(os.path.join("results",f"{date_now}_" + "CandelsSum.csv")):
  dfResults = pd.read_csv(os.path.join("results", f"{date_now}_" + "CandelsSum.csv"))
  print(dfResults[["Date","Close","ticker","candlestick_pattern","UpOrDown"]])
  dfPer=dfResults['UpOrDown'].value_counts(normalize=True) * 100
  print(dfPer)
