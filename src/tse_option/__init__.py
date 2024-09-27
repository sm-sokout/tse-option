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
    html = rq.get(url, headers=headers).text
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
    tsetmc = rq.get(url)
    return int(tsetmc.text.split(";")[0].split(",")[2])

#----------------------------------------

def tse_options(symbol, stock=True):
    url = "https://old.tse.ir/json/MarketWatch/MarketWatch_7.xml"
    headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36'}
    data = pd.read_html(url)[0]
    data = data[data["نماد"].notna()]
    if stock == True:
        data["نماد دارایی پایه"] = [data.iloc[i]['نام'].split("-")[0].split(" ")[1] for i in range(len(data))]
        data = data[data["نماد دارایی پایه"] == symbol]
    else:
        data = data[data["نماد"] == symbol]
    return data

#----------------------------------------
'''
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
'''
#----------------------------------------


def download(symbols, j_date=False, start=None, end=None, adjust_price=False, drop_unadjusted=False):
    headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36'}
    if type(symbols) == str:
        symbols = [symbols]
    for symbol in symbols:
        symbol = symbol.replace("ی","ي").replace('ک','ك')
        id = stock_id(symbol)
        url = 'http://old.tsetmc.com/tsev2/data/Export-txt.aspx?t=i&a=1&b=0&i=' + id
        html=rq.get(url,headers=headers).content
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
    last_price = stock_price(symbol)
    df_stock = download(symbol, start=None, end=None, adjust_price=True)[-trading_days:]
    df = df_stock[["Adj Close"]].copy()
    df["return"] = np.log(df["Adj Close"]/df["Adj Close"].shift())
    sigma = df["return"].std() * np.sqrt(240)
    try:
        rf = risk_free_interest_rate()
    except:
        print('در حال حاضر دریافت اطلاعات از فرابورس امکان پذیر نیست.\nلطفا نرخ بهره بدون ریسک مورد نظر را بصورت یک عدد اعشاری وارد کنید.')
        rf = float(input('... '))
    data = tse_options(symbol)
    if data.empty:
        #data = farabourse_options(symbol)
        raise NameError('در حال حاضر دریافت اطلاعات از فرابورس امکان پذیر نیست.')
    data.reset_index(inplace=True, drop=True)
    col = ["Symbol", "Strike Price", "Status", "Maturity(day)","Open Interest", "Bid", "Ask", "Delta"]
    if IV == True:
        col.append("IV")
    if leverage == True:
        col.append("Leverage")
    col.append("BSM")
    if P_BSM == True:
        col.append("%Price/BSM")
    df = pd.DataFrame(index=range(len(data)),columns=col)

    for i in range(len(data)):
        df.loc[i,'Symbol'] = data.loc[i,"نماد"]
        df.loc[i,"Stock"] = symbol
        df.loc[i,"Strike Price"] = int(data.loc[i,"قیمت اعمال"])
        if last_price > 1.025*df.loc[i,"Strike Price"]:
            df.loc[i,'Status'] = "ITM"
        elif last_price < 0.975*df.loc[i,"Strike Price"]:
            df.loc[i,'Status'] = "OTM"
        else:
            df.loc[i,'Status'] = "ATM"
        df.loc[i,"Maturity(day)"] = int(data.loc[i,"روزهای باقیمانده تا سررسید"])
        df.loc[i,'Open Interest'] = data.loc[i,"موقیعت های باز"]
        df.loc[i,'Bid'] = data.loc[i,"حجم بهترین سفارش خرید"]
        df.loc[i,'Ask'] = data.loc[i,"قیمت بهترین سفارش فروش"]
 

    if sort.upper() == "MATURITY":
        df = df.sort_values(by=['Maturity(day)'], ascending=True)
    elif sort.upper() == "STRIKE PRICE":
        df = df.sort_values(by=['Strike Price'], ascending=True)
    else:
        df = df.sort_values(by=['Open Interest'], ascending=False)
    df = df.reset_index(drop=True)
    #return df
    for i in range(len(df)):
        K = df.loc[i,"Strike Price"]
        T = df.loc[i,"Maturity(day)"]/360
        d1 = (np.log(last_price/K) + (rf + (sigma ** 2 )/2) * T)/(sigma * (T ** 0.5))
        d2 = d1 - (sigma * (T ** 0.5))
        df.loc[i, 'BSM'] = round((last_price * norm.cdf(d1)) - (K * np.exp(-rf*T) * norm.cdf(d2)))
        market_price = int(df.loc[i,'Ask'])
        df.loc[i,'Delta'] = round(norm.cdf(d1),2)
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
    return df

#----------------------------------------

def call(option_symbol:str, trading_days:int=100, IV=False, leverage=True, P_BSM=False):
    """
    Valuation of a particular option
    """
    option_symbol = option_symbol.replace("ی","ي").replace('ک','ك')
    headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36'}
    url = f"http://old.tsetmc.com/tsev2/data/search.aspx?skey={option_symbol}"
    html = rq.get(url, headers=headers).text
    if len(html.split(";")) == 2:
        stock_symbol = html.split(",")[1].split("-")[0].split(" ")[1]
        last_price = stock_price(stock_symbol)
        df_stock = download(stock_symbol, start=None, end=None, adjust_price=True)[-trading_days:]
        df = df_stock[["Adj Close"]].copy()
        df["return"] = np.log(df["Adj Close"]/df["Adj Close"].shift())
        sigma = df["return"].std() * np.sqrt(240)
        rf = risk_free_interest_rate()
        option_in_tse = option_symbol + "1"
        data = tse_options(option_in_tse, False)
        if data.empty:
            #data = farabourse_options(option_symbol, False)
            data = pd.DataFrame()
        data.reset_index(inplace=True, drop=True)
        if data.empty:
            raise NameError("This symbol does not exist or has expired. Please enter the symbol correctly!")
        
        else:
            col = ["Symbol", "Strike Price", "Status", "Maturity(day)","Open Interest", "Bid", "Ask", "Delta"]
            if IV == True:
                col.append("IV")
            if leverage == True:
                col.append("Leverage")
            col.append("BSM")
            if P_BSM == True:
                col.append("%Price/BSM")
            df = pd.DataFrame(index=range(len(data)),columns=col)

            df.loc[0,'Symbol'] = data.loc[0,"نماد"]
            df.loc[0,'Stock'] = stock_symbol
            df.loc[0,'Strike Price'] = data.loc[0,"قیمت اعمال"]
            if last_price > 1.025*df.loc[0,"Strike Price"]:
                df.loc[0,"Status"] = "ITM"
            elif last_price < 0.975*df.loc[0,"Strike Price"]:
                df.loc[0,"Status"] = "OTM"
            else:
                df.loc[0,"Status"] = "ATM"
            df.loc[0,"Maturity(day)"] = int(data.loc[0,"روزهای باقیمانده تا سررسید"])
            df.loc[0,"Open Interest"] = data.loc[0,"موقیعت های باز"]
            df.loc[0,"Bid"] = data.loc[0,"حجم بهترین سفارش خرید"]
            df.loc[0,"Ask"] = data.loc[0,"قیمت بهترین سفارش فروش"]

            df = df.reset_index(drop=True)

            K = df.loc[0,"Strike Price"]
            T = df.loc[0,"Maturity(day)"]/360
            d1 = (np.log(last_price/K) + (rf + (sigma ** 2 )/2) * T)/(sigma * (T ** 0.5))
            d2 = d1 - (sigma * (T ** 0.5))
            df.loc[0,"BSM"] = round((last_price * norm.cdf(d1)) - (K * np.e ** (-rf*T) * norm.cdf(d2)))
            market_price = int(df.loc[0,"Ask"])
            df.loc[0,"Delta"] = round(norm.cdf(d1),2)
            if IV == True:
                if market_price == 0:
                    df.loc[0,"IV"] = "%-"
                else:
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
            return df


   


    elif len(html.split(";")) == 1:
        raise NameError("This symbol does not exist. Please enter the symbol correctly!")
    else:
        raise NameError("Please enter the symbol completely!")






