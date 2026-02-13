#Importing
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import requests as rq
from scipy.stats import norm
import jdatetime
import jalali_pandas
import io
import math
import json
import time

#----------------------------------------

def BSM_call(S0, K, rf, T, sigma):
    """
    Black, Scholes and Merton model for option pricing
    """
    d1 = (np.log(S0/K) + (rf + (sigma ** 2 )/2) * T)/(sigma * (T ** 0.5))
    d2 = d1 - (sigma * (T ** 0.5))
    call_price = (S0 * norm.cdf(d1)) - (K * np.exp(-rf*T) * norm.cdf(d2))
    return call_price

#----------------------------------------

def BSM_put(S0, K, rf, T, sigma):
    
    d1 = (np.log(S0 / K) + (rf + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    put_price = K * np.exp(-rf * T) * norm.cdf(-d2) - S0 * norm.cdf(-d1)
    return put_price

#----------------------------------------

def delta(S0, K, rf, T, sigma, type):
    d1 = (np.log(S0/K) + (rf+0.5*sigma**2)*T) / (sigma*np.log(T))
    if type == 'call':
        return norm.cdf(d1)
    elif type == 'put':
        return norm.cdf(d1) - 1

#----------------------------------------

def Vega(S0, K, rf, T, Sigma):
    d1 = (np.log(S0 / K) + (rf + 0.5 * Sigma ** 2) * T) / (Sigma * np.sqrt(T))
    return S0 * norm.pdf(d1) * np.sqrt(T)

#----------------------------------------

def find_IV(market_price, S0, K, rf, T, *args):
    """
    Calculation of Implied Volatility
    """
    MAX_ITERATIONS = 200
    PRECISION = 1.0e-4
    Sigma = 0.5
    for i in range(0, MAX_ITERATIONS):
        price = BSM_call(S0, K, rf, T, Sigma)
        vega = Vega(S0, K, rf, T, Sigma)
        diff = market_price - price  # our root
        if (abs(diff) < PRECISION):
            return Sigma
        Sigma = Sigma + diff/vega # f(x) / f'(x)
    return Sigma

#----------------------------------------

def risk_free_interest_rate():
    """"
    Calculation of risk-free interest rate based on the yield of treasury bills
    """
    pd.options.mode.copy_on_write = True 
    url ='https://ifb.ir/ytm.aspx'
    df = pd.read_html(url, encoding='utf-8')[1]
    df = df[df['YTM'] != 'سررسید شده']
    df['YTM'] = df['YTM'].str.replace('/', '.')
    df['YTM'] = df['YTM'].str.rstrip('%')
    df['YTM'] = df['YTM'].astype(float)
    rf = df['YTM'].mean()/100
    return rf

#----------------------------------------

def stock_id(symbol):
    headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36'}
    url = f"http://old.tsetmc.com/tsev2/data/search.aspx?skey={symbol}"
    html = rq.get(url, headers=headers,verify=False).text
    info = html.split(";")
    for i in info:
        if i.split(",")[0] == symbol:
            return i.split(",")[2]
        else:
            try:
                return info[0].split(",")[2]
            except:
                raise NameError("This symbol does not exist. Please enter the symbol correctly!")

#----------------------------------------

def stock_price(symbol):
    id = stock_id(symbol)
    if id is None:
        raise NameError("This symbol does not exist. Please enter the symbol correctly!")
    url = "http://old.tsetmc.com/tsev2/data/instinfofast.aspx?i="+id+"&c=34"
    headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36'}
    tsetmc = rq.get(url, headers=headers, verify=False)
    return int(tsetmc.text.split(";")[0].split(",")[2])

#----------------------------------------
"""
def tse_call(symbol, stock=True):
    url = "https://old.tse.ir/json/MarketWatch/MarketWatch_7.xml"
    headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36'}
    data = pd.read_html(url)[0]
    data = data[data["نماد"].notna()]
    if stock == True:
        data["نماد دارایی پایه"] = [data.iloc[i]['نام'].split("-")[0].split(" ")[1] for i in range(len(data))]
        data = data[data["نماد دارایی پایه"] == symbol]
    else:
        if symbol in data["نماد"].values:
            data = data[data["نماد"] == symbol]
        else:
            data = data[data["نماد"] == symbol[:-1]]
    return data

#----------------------------------------

def tse_put(symbol, stock=True):
    url = "https://old.tse.ir/json/MarketWatch/MarketWatch_7.xml"
    headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36'}
    data = pd.read_html(url)[0]
    data = data[data["نماد"].notna()]
    data['نماد'] = data['نماد'].apply(lambda x: 'ط'+x[1:])
    data['نام'] = data['نام'].apply(lambda x: x.replace('اختيارخ','اختيارف'))
    if stock == True:
        data["نماد دارایی پایه"] = [data.iloc[i]['نام'].split("-")[0].split(" ")[1] for i in range(len(data))]
        data = data[data["نماد دارایی پایه"] == symbol]
    else:
        if symbol in data["نماد"].values:
            data = data[data["نماد"] == symbol]
        else:
            data = data[data["نماد"] == symbol[:-1]]
    return data[['نماد','نام','موقیعت های باز','اندازه قرارداد','روزهای باقیمانده تا سررسید','قیمت اعمال','قیمت مبنای دارایی پایه']]

#----------------------------------------

def farabourse_options(symbol, stock=True):
    url = "https://ifb.ir/OptionStockQuote.aspx"
    headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36'}
    fopen = rq.get(url,headers=headers).content
    html = pd.read_html(io.StringIO(fopen.decode("utf-8")), header=2)
    data = pd.DataFrame(html[0])
    if stock == True:
        data = data[data['نماد دارایی پایه'] == symbol]
    else:
        data = data[data['نماد'] == symbol]
    data = data.rename(columns={"روزهای باقی‌مانده تا سررسید": "روزهای باقیمانده تا سررسید",
    "قیمت.1": "حجم بهترین سفارش خرید","قیمت":"قیمت بهترین سفارش فروش",
    "تعداد موقعیت\u200cهای باز":"موقیعت های باز"})
    return data
"""
#----------------------------------------

def option_reader(market):
    if market.lower() == 'tse':
        url = 'https://cdn.tsetmc.com/api/Instrument/GetInstrumentOptionMarketWatch/1'
        market = 'bourse'
    else:
        url = 'https://cdn.tsetmc.com/api/Instrument/GetInstrumentOptionMarketWatch/2'
        market = 'farabourse'
    headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36'}
    html = rq.get(url, headers=headers, timeout=10, verify=False)
    df = json.loads(html.text)
    df = pd.DataFrame(df['instrumentOptMarketWatch'])
    df['beginDate'] = df['beginDate'].apply(lambda x: x[:4]+'-'+x[4:6]+'-'+x[6:])
    df['endDate'] = df['endDate'].apply(lambda x: x[:4]+'-'+x[4:6]+'-'+x[6:])
    df = df.rename(columns={'lval30_UA':'UA','pClosing_UA':'close_UA','priceYesterday_UA':'yesterday_UA'})

    df = df.rename(columns={'pDrCotVal_C':'last_C','oP_C':'oi_C','pClosing_C':'final_C','priceYesterday_C':'yesterday_C','qTotCap_C':'value_C','qTotTran5J_C':'volume_C',\
                            'zTotTran_C':'numTrans_C','lVal30_C':'description_C','lVal18AFC_C':'symbol_C','pMeDem_C':'bidP_C','qTitMeDem_C':'bidV_C','pMeOf_C':'askP_C',\
                            'qTitMeOf_C':'askV_C','yesterdayOP_C':'yestOI_C'})

    df = df.rename(columns={'pDrCotVal_P':'last_P','oP_P':'oi_P','pClosing_P':'final_P','priceYesterday_P':'yesterday_P','qTotCap_P':'value_P','qTotTran5J_P':'volume_P',\
                            'zTotTran_P':'numTrans_P','lVal30_P':'description_P','lVal18AFC_P':'symbol_P','pMeDem_P':'bidP_P','qTitMeDem_P':'bidV_P','pMeOf_P':'askP_P',\
                            'qTitMeOf_P':'askV_P','yesterdayOP_P':'yestOI_P'})
    df['UA'] = df['description_C'].apply(lambda x: x.split('-')[0].split('اختيارخ ')[-1])
    df['market'] = market
    col_general = ['contractSize','beginDate','endDate','strikePrice','remainedDay','market']
    col_ua = ['UA','close_UA','yesterday_UA']
    col_c = ['last_C','oi_C','final_C','yesterday_C','value_C','volume_C','numTrans_C','description_C','symbol_C','bidP_C','bidV_C','askP_C','askV_C','yestOI_C']
    col_p = ['last_P','oi_P','final_P','yesterday_P','value_P','volume_P','numTrans_P','description_P','symbol_P','bidP_P','bidV_P','askP_P','askV_P','yestOI_P']

    call = df[col_c+col_ua+col_general].copy()
    put = df[col_p+col_ua+col_general].copy()

    call = call.rename(columns={'symbol_C':'ticker','description_C':'description','oi_C':'OI','volume_C':'volume','value_C':'value','numTrans_C':'transaction',\
                                'last_C':'last_price','askV_C':'ask_vol','askP_C':'ask','bidP_C':'bid','bidV_C':'bid_vol','contractSize':'size','remainedDay':'TTM',\
                                'strikePrice':'exc_price','close_UA':'underlying_last','endDate':'maturity','final_C':'final','yesterday_C':'yesterday','yestOI_C':'yesterday_OI'})

    put = put.rename(columns={'symbol_P':'ticker','description_P':'description','oi_P':'OI','volume_P':'volume','value_P':'value','numTrans_P':'transaction',\
                                'last_P':'last_price','askV_P':'ask_vol','askP_P':'ask','bidP_P':'bid','bidV_P':'bid_vol','contractSize':'size','remainedDay':'TTM',\
                                'strikePrice':'exc_price','close_UA':'underlying_last','endDate':'maturity','final_P':'final','yesterday_P':'yesterday','yestOI_P':'yesterday_OI'})
    call = call[['ticker','description','OI','volume','value','last_price','ask_vol','ask','bid','bid_vol','size','TTM',\
                 'exc_price','underlying_last','maturity','final','UA','market']]

    put = put[['ticker','description','OI','volume','value','last_price','ask_vol','ask','bid','bid_vol','size','TTM',\
               'exc_price','underlying_last','maturity','final','UA','market']]
    return call, put

#----------------------------------------

def initial_margin(S:int, K:int, premium:int, size:int=1000, type:str="call")->int:
    """
    this function is used to calculation of initial margin
    
    input--> 1. S:       underlying asset price
             2. K:       strike price
             3. premium: option price in the market
             4. size:    the size of option contract.
             5. type:    type of the option
    
    output--> the amount of margin that we have to pay at the beginning.
    """
    round_factor = 100_000
    if type == "call":
        L = abs(np.minimum(S-K, 0)*size)
    else:
        L = abs(np.minimum(K-S, 0)*size)
    A = 0.2*S*size - L
    B = 0.1*K*size
    try:
        margin = (((math.floor(np.maximum(A,B)/round_factor))+1)*round_factor)
        return margin+premium*size
    except:
        #print(S,K,premium,size,type)
        return np.NaN

#----------------------------------------

def download(symbols, j_date=False, start=None, end=None, adjust_price=False, drop_unadjusted=False):
    headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36'}
    if type(symbols) == str:
        symbols = [symbols]
    for symbol in symbols:
        symbol = symbol.replace("ی","ي").replace('ک','ك')
        id = stock_id(symbol)
        url = 'http://old.tsetmc.com/tsev2/data/Export-txt.aspx?t=i&a=1&b=0&i=' + id
        html=rq.get(url,headers=headers,verify=False).content
        df=pd.read_csv(io.StringIO(html.decode("utf-8")),index_col="<DTYYYYMMDD>",parse_dates=True)[::-1]
        df=df.rename(columns={"<OPEN>": "Yesterday", "<CLOSE>": "Close","<FIRST>":"Open","<HIGH>":"High","<LOW>":"Low","<VOL>":"Volume"})
        df.index.rename('Date',inplace=True)
        df = df[["Open","High","Low","Close","Volume","Yesterday"]]
        
        def adjusting(df:pd.DataFrame)->pd.DataFrame:
            df["temp"] = (df["Close"].shift(1) / df["Yesterday"]).fillna(1)
            df["temp"] = (df["temp"].iloc[::-1].cumprod().iloc[::-1].shift(-1)).fillna(1)
            df["Adj Open"] = (df["Open"] / df["temp"]).astype(int)
            df["Adj High"] = (df["High"] / df["temp"]).astype(int)
            df["Adj Low"] = (df["Low"] / df["temp"]).astype(int)
            df["Adj Close"] = (df["Close"] / df["temp"]).astype(int)
            df = df.drop(columns=["temp"])
            return df
        if adjust_price:
            df = adjusting(df)
            if drop_unadjusted:
                df["Open"] = df["Adj Open"]
                df["High"] = df["Adj High"]
                df["Low"] = df["Adj Low"]
                df["Close"] = df["Adj Close"]
                df.drop(["Adj Open","Adj High","Adj Low","Adj Close"], axis=1, inplace=True)
        df.drop(["Yesterday"], axis=1, inplace=True) 
        if len(symbols) > 1:
            for j in df.columns:
                df = df.rename(columns={f'{j}': f'{symbol},{j}'})
            try:
                data = pd.concat([data, df],axis=1)
            except:
                data = df.copy()
    if len(symbols) > 1:
        data.columns = pd.MultiIndex.from_tuples([(x.split(",")[1], x.split(",")[0]) for x in data.columns])
        data["Date","Date"] = data.index
        data["JDate"] = data["Date","Date"].jalali.to_jalali()
        data = data.reindex(data.columns.levels[0], level=0, axis=1)
    else:
        data = df.copy()
        data["Date"] = data.index
        data["JDate"] = data["Date"].jalali.to_jalali()
    if j_date:
        if start != None:
            start = jdatetime.date(int(start[:4]),int(start[5:7]),int(start[8:])).togregorian()
            start = str(start.year)+"-"+str(start.month) + "-" + str(start.day)
        if end != None:
            end = jdatetime.date(int(end[:4]),int(end[5:7]),int(end[8:])).togregorian()
            end = str(end.year)+"-"+str(end.month)+"-"+str(end.day)
    if start != None:
        data = data[start:]
    if end != None:
        data = data[:end]
    data.drop(["Date"], axis=1, inplace=True)
    return data

#----------------------------------------

def option_chain(symbol:str, trading_days:int=100, IV=False, leverage=True, P_BSM=False, sort="Maturity"):
    """
    Valuation of all options on a particular stock
    """
    symbol = symbol.replace("ی","ي").replace('ک','ك')
    #last_price = stock_price(symbol)
    df_stock = download(symbol, start=None, end=None, adjust_price=True)[-trading_days:]
    df = df_stock[["Adj Close"]].copy()
    df["return"] = np.log(df["Adj Close"]/df["Adj Close"].shift())
    sigma = df["return"].std() * np.sqrt(240)
    try:
        rf = risk_free_interest_rate()
    except:
        print('در حال حاضر دریافت اطلاعات از فرابورس امکان پذیر نیست.\nلطفا نرخ بهره بدون ریسک مورد نظر را بصورت یک عدد اعشاری وارد کنید.')
        rf = float(input('... '))
    option_watch, _ = option_reader('tse')
    data = option_watch[option_watch['UA']==symbol]
    #data = tse_call(symbol)
    if len(data)==0:
        option_watch, _ = option_reader('ifb')
        data = option_watch[option_watch['UA']==symbol]
        if len(data)==0:
            raise NameError('نماد دارایی پایه را به درستی وارد کنید.')
            
    data.reset_index(inplace=True, drop=True)
    last_price = int(data.iloc[0]['underlying_last'])
    col = ["Symbol", "Strike Price", "Status", "Maturity(day)", "Size", "Open Interests", "Bid", "Ask", "Initial Margin", "Delta"]
    if IV == True:
        col.append("IV")
    if leverage == True:
        col.append("Leverage")
    col.append("BSM")
    if P_BSM == True:
        col.append("%Price/BSM")
    df = pd.DataFrame(index=range(len(data)),columns=col)

    for i in range(len(data)):
        df.loc[i,'Symbol'] = data.loc[i,"ticker"]
        df.loc[i,"Stock"] = symbol
        df.loc[i,"Strike Price"] = int(data.loc[i,"exc_price"])
        if last_price > 1.025*df.loc[i,"Strike Price"]:
            df.loc[i,'Status'] = "ITM"
        elif last_price < 0.975*df.loc[i,"Strike Price"]:
            df.loc[i,'Status'] = "OTM"
        else:
            df.loc[i,'Status'] = "ATM"
        df.loc[i,"Maturity(day)"] = int(data.loc[i,"TTM"])
        df.loc[i,"Size"] = data.loc[i,"size"]
        df.loc[i,'Open Interests'] = data.loc[i,"OI"]
        df.loc[i,'Bid'] = data.loc[i,"bid"]
        df.loc[i,'Ask'] = data.loc[i,"ask"]
 

    if sort.upper() == "MATURITY":
        df = df.sort_values(by=['Maturity(day)'], ascending=True)
    elif sort.upper() == "STRIKE PRICE":
        df = df.sort_values(by=['Strike Price'], ascending=True)
    else:
        df = df.sort_values(by=['Open Interest'], ascending=False)
    df = df.reset_index(drop=True)

    df['Initial Margin'] = df.apply(lambda x: initial_margin(last_price, x['Strike Price'], x['Bid'], x["Size"], 'call'), axis=1)
    df['BSM'] = df.apply(lambda x: round(BSM_call(last_price, x['Strike Price'], rf, x['Maturity(day)']/360, sigma)), axis=1)
    df['Delta'] = df.apply(lambda x: round(delta(last_price, x['Strike Price'], rf, x['Maturity(day)']/360, sigma, 'call'),2), axis=1)
    
    for i in range(len(df)):
        K = df.loc[i,"Strike Price"]
        T = df.loc[i,"Maturity(day)"]/360
        market_price = int(df.loc[i,'Ask'])
        if IV == True:
            if market_price == 0:
                df.loc[i,'IV'] = "%-"
            else:
                df.loc[i,'IV'] = f"%{round(find_IV(market_price, last_price, K, rf, T)*100,1)}"
                if df.loc[i,'IV']=='%nan':
                    df.loc[i,'IV'] = "%-"
            

        if leverage == True:
            try:
                df.loc[i,'Leverage'] = round(df.loc[i,'Delta'] * last_price / df.loc[i,'Ask'], 2)
            except:
                df.loc[i,'Leverage'] = np.inf
        if P_BSM == True:
            try:
                price_bsm = round(-100*(market_price/df.loc[i,"BSM"]-1),1)
                
                if price_bsm >=0:
                    df.loc[i,'%Price/BSM'] = f"%{price_bsm}\U0001f7e2"
                else:
                    df.loc[i,'%Price/BSM'] = f"%{price_bsm}\U0001F534"
                if market_price == 0:
                    df.loc[i,'%Price/BSM'] = "%-"
            except:
                df.loc[i,'%Price/BSM'] = "%-"


    df.index = df["Symbol"]
    df = df.drop("Symbol", axis=1)
    print(f"Stock Price: {int(last_price)} \tRiskFreeRate: {round(rf*100,2)}% \tHV: {round(sigma*100,2)}%")
    print('\nوجه تضمین به ازای هر قرارداد محاسبه شده است.')
    return df

#----------------------------------------

def call(option_symbol:str, trading_days:int=100, IV=False, leverage=True, P_BSM=False):
    """
    Valuation of a particular option
    """
    option_symbol = option_symbol.replace("ی","ي").replace('ک','ك')
    #headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36'}
    #url = f"http://old.tsetmc.com/tsev2/data/search.aspx?skey={option_symbol}"
    #html = rq.get(url, headers=headers).text
    #if len(html.split(";")) == 2:
    option_watch, _ = option_reader('tse')
    data = option_watch[option_watch['ticker']==option_symbol]
    if len(data)==0:
        option_watch, _ = option_reader('ifb')
        data = option_watch[option_watch['ticker']==option_symbol]
        if len(data)==0:
            raise NameError('این نماد اختیار معامله وجود ندارد و یا اینکه سررسید شده است.')
    data.reset_index(inplace=True, drop=True)
    stock_symbol = data.iloc[0]['UA']
    #last_price = stock_price(stock_symbol)
    df_stock = download(stock_symbol, start=None, end=None, adjust_price=True)[-trading_days:]
    df = df_stock[["Adj Close"]].copy()
    df["return"] = np.log(df["Adj Close"]/df["Adj Close"].shift())
    sigma = df["return"].std() * np.sqrt(240)
    try:
        rf = risk_free_interest_rate()
    except:
        print('در حال حاضر دریافت اطلاعات از فرابورس امکان پذیر نیست.\nلطفا نرخ بهره بدون ریسک مورد نظر را بصورت یک عدد اعشاری وارد کنید.')
        rf = float(input('... '))
    last_price = int(data.iloc[0]['underlying_last'])
    col = ["Symbol", "Strike Price", "Status", "Maturity(day)", "Size", "Open Interests", "Bid", "Ask", "Last Price", "Initial Margin", "Delta"]
    if IV == True:
        col.append("IV")
    if leverage == True:
        col.append("Leverage")
    col.append("BSM")
    if P_BSM == True:
        col.append("%Price/BSM")
    df = pd.DataFrame(index=range(len(data)),columns=col)

    df.loc[0,'Symbol'] = data.loc[0,"ticker"]
    df.loc[0,'Stock'] = stock_symbol
    df.loc[0,'Strike Price'] = data.loc[0,"exc_price"]
    if last_price > 1.025*df.loc[0,"Strike Price"]:
        df.loc[0,"Status"] = "ITM"
    elif last_price < 0.975*df.loc[0,"Strike Price"]:
        df.loc[0,"Status"] = "OTM"
    else:
        df.loc[0,"Status"] = "ATM"
    df.loc[0,"Maturity(day)"] = int(data.loc[0,"TTM"])
    df.loc[0,"Size"] = data.loc[0,"size"]
    df.loc[0,"Open Interests"] = data.loc[0,"OI"]
    df.loc[0,"Bid"] = data.loc[0,"bid"]
    df.loc[0,"Ask"] = data.loc[0,"ask"]
    df.loc[0,"Last Price"] = data.loc[0,"last_price"] #stock_price(option_symbol)

    df = df.reset_index(drop=True)

    df['Initial Margin'] = df.apply(lambda x: initial_margin(last_price, x['Strike Price'], x['Bid'], x["Size"], 'call'), axis=1)
    df['BSM'] = df.apply(lambda x: round(BSM_call(last_price, x['Strike Price'], rf, x['Maturity(day)']/360, sigma)), axis=1)
    df['Delta'] = df.apply(lambda x: round(delta(last_price, x['Strike Price'], rf, x['Maturity(day)']/360, sigma, 'call'),2), axis=1)

    market_price = int(df.loc[0,"Last Price"])

    if IV == True:
        if market_price == 0:
            df.loc[0,"IV"] = "%-"
        else:
            K = df.loc[0, "Strike Price"]
            T = df.loc[0, "Maturity(day)"]/360
            df.loc[0,"IV"] = f"%{round(find_IV(market_price, last_price, K, rf, T)*100,1)}"
            if df.loc[0,"IV"] == '%nan':
                df.loc[0,"IV"] = "%-"
    if leverage == True:
        try:
            df.loc[0,"Leverage"] = round(df.loc[0,"Delta"] * last_price /market_price, 2)
        except:
            df.loc[0,"Leverage"] = np.inf

    if P_BSM == True:
        try:
            price_bsm = round(-100*(market_price/df.loc[0,"BSM"]-1),1)
            if price_bsm >=0:
                df.loc[0,"%Price/BSM"] = f"%{price_bsm}\U0001f7e2"
            else:
                df.loc[0,"%Price/BSM"] = f"%{price_bsm}\U0001F534"
            if market_price == 0:
                df.loc[0,"%Price/BSM"] = "%-"
        except:
            df.loc[0,"%Price/BSM"] = "%-"

    df.index = df["Symbol"]
    df = df.drop("Symbol", axis=1)
    print(f"Stock Price: {int(last_price)}\tRiskFreeRate: {round(rf*100,2)}%\tHV: {round(sigma*100,2)}%")
    print('\nوجه تضمین به ازای هر قرارداد محاسبه شده است.')
    return df

#----------------------------------------

def put(option_symbol:str, trading_days:int=100, leverage=False, P_BSM=False):
    option_symbol = option_symbol.replace("ی","ي").replace('ک','ك')
    #headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36'}
    #url = f"http://old.tsetmc.com/tsev2/data/search.aspx?skey={option_symbol}"
    #html = rq.get(url, headers=headers).text
    #if len(html.split(";")) == 2:
    _, option_watch = option_reader('tse')
    data = option_watch[option_watch['ticker']==option_symbol]
    if len(data)==0:
        _, option_watch = option_reader('ifb')
        data = option_watch[option_watch['ticker']==option_symbol]
        if len(data)==0:
            raise NameError('این نماد اختیار معامله وجود ندارد و یا اینکه سررسید شده است.')
    data.reset_index(inplace=True, drop=True)
    stock_symbol = data.iloc[0]['UA']
    df_stock = download(stock_symbol, start=None, end=None, adjust_price=True)[-trading_days:]
    df = df_stock[["Adj Close"]].copy()
    df["return"] = np.log(df["Adj Close"]/df["Adj Close"].shift())
    sigma = df["return"].std() * np.sqrt(240)
    try:
        rf = risk_free_interest_rate()
    except:
        print('در حال حاضر دریافت اطلاعات از فرابورس امکان پذیر نیست.\nلطفا نرخ بهره بدون ریسک مورد نظر را بصورت یک عدد اعشاری وارد کنید.')
        rf = float(input('... '))
    last_price = int(data.iloc[0]['underlying_last'])
    col = ["Symbol", "Strike Price", "Status", "Maturity(day)", "Size", "Open Interests", "Bid", "Ask", "Last Price", "Initial Margin", "Delta"]
    if leverage == True:
        col.append("Leverage")
    col.append("BSM")
    if P_BSM == True:
        col.append("%Price/BSM")
    df = pd.DataFrame(index=range(len(data)),columns=col)

    df.loc[0,'Symbol'] = data.loc[0,"ticker"]
    df.loc[0,'Stock'] = stock_symbol
    df.loc[0,'Strike Price'] = data.loc[0,"exc_price"]
    if last_price > 1.025*df.loc[0,"Strike Price"]:
        df.loc[0,"Status"] = "OTM"
    elif last_price < 0.975*df.loc[0,"Strike Price"]:
        df.loc[0,"Status"] = "ITM"
    else:
        df.loc[0,"Status"] = "ATM"
    df.loc[0,"Maturity(day)"] = int(data.loc[0,"TTM"])
    df.loc[0,"Size"] = data.loc[0,"size"]
    df.loc[0,"Open Interests"] = data.loc[0,"OI"]
    df.loc[0,"Bid"] = data.loc[0,"bid"]
    df.loc[0,"Ask"] = data.loc[0,"ask"]
    df.loc[0,"Last Price"] = data.loc[0,"last_price"]

    df = df.reset_index(drop=True)
    df['Initial Margin'] = df.apply(lambda x: initial_margin(last_price, x['Strike Price'], x['Last Price'], x["Size"], 'put'), axis=1)
    df['BSM'] = df.apply(lambda x: round(BSM_put(last_price, x['Strike Price'], rf, x['Maturity(day)']/360, sigma)), axis=1)
    df['Delta'] = df.apply(lambda x: round(delta(last_price, x['Strike Price'], rf, x['Maturity(day)']/360, sigma, 'put'),2), axis=1)
    
    market_price = int(df.loc[0,"Last Price"])

    if leverage == True:
        try:
            df.loc[0,"Leverage"] = round((-df.loc[0,"Delta"]) * last_price /market_price, 2)
        except:
            df.loc[0,"Leverage"] = np.inf

    if P_BSM == True:
        try:
            price_bsm = round(-100*(market_price/df.loc[0,"BSM"]-1),1)
            if price_bsm >=0:
                df.loc[0,"%Price/BSM"] = f"%{price_bsm}\U0001f7e2"
            else:
                df.loc[0,"%Price/BSM"] = f"%{price_bsm}\U0001F534"
            if market_price == 0:
                df.loc[0,"%Price/BSM"] = "%-"
        except:
            df.loc[0,"%Price/BSM"] = "%-"

    df.index = df["Symbol"]
    df = df.drop("Symbol", axis=1)
    print(f"Stock Price: {int(last_price)}\tRiskFreeRate: {round(rf*100,2)}%\tHV: {round(sigma*100,2)}%")
    print('\nوجه تضمین به ازای هر قرارداد محاسبه شده است.')
    return df






