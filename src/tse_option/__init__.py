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
    call_price = (S0 * norm.cdf(d1)) - (K * np.e ** (-rf*T) * norm.cdf(d2))
    return call_price

#----------------------------------------

def bs_vega(S0, K, rf, T, Sigma):
    d1 = (np.log(S0 / K) + (rf + 0.5 * Sigma ** 2) * T) / (Sigma * np.sqrt(T))
    return S0 * norm.pdf(d1) * np.sqrt(T)

#----------------------------------------


def find_IV(market_price, S0, K, rf, T, *args):
    """
    Calculation of Impolied Volatility
    """
    MAX_ITERATIONS = 200
    PRECISION = 1.0e-4
    Sigma = 0.5
    for i in range(0, MAX_ITERATIONS):
        price = BSM_call(S0, K, rf, T, Sigma)
        vega = bs_vega(S0, K, rf, T, Sigma)
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
    url = "https://ifb.ir/ytm.aspx"
    headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36'}
    fopen = rq.get(url, headers=headers).content
    html = pd.read_html(io.StringIO(fopen.decode("utf-8")), header=0, index_col=0)
    html = pd.DataFrame(html[1])
    html = html["YTM"].replace({"سررسید شده":np.nan})
    html = html.dropna().reset_index(drop=True)
    ytm = []
    for i in html:
        if len(i.split("%")[0].split("/")) == 1:
            ytm.append(int(i.split("%")[0].split("/")[0])/100)
        elif len(i.split("%")[0].split("/")[0])==1:
            ret = i.split("%")[0].split("/")[0] + i.split("%")[0].split("/")[1]
            ytm.append(int(ret)/10**(len(list(ret))+1)) 
        else:
            ret = i.split("%")[0].split("/")[0] + i.split("%")[0].split("/")[1]
            ytm.append(int(ret)/10**(len(list(ret))))          
            
    return np.mean(ytm)

#----------------------------------------

def stock_id(stock_name):
    headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36'}
    url = f"http://www.tsetmc.com/tsev2/data/search.aspx?skey={stock_name}"
    html = rq.get(url, headers=headers).text
    info = html.split(";")
    for i in info:
        if i.split(",")[0] == stock_name:
            return i.split(",")[2]
        else:
            try:
                return info[0].split(",")[2]
            except:
                raise NameError("This symbol does not exist. Please enter the symbol correctly!")

#----------------------------------------

def stock_price(stock_name):
    id = stock_id(stock_name)
    if id is None:
        raise NameError("This symbol does not exist. Please enter the symbol correctly!")
    url = "http://www.tsetmc.com/tsev2/data/instinfofast.aspx?i="+id+"&c=34"
    tsetmc = rq.get(url)
    return int(tsetmc.text.split(";")[0].split(",")[2])

#----------------------------------------

def tse_options(symbol, stock=True):
    url = "https://tse.ir/json/MarketWatch/MarketWatch_7.xml"
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

#----------------------------------------


def download_history(symbol, j_date=False, start=None, end=None, adjust_price=False, drop_unadjusted=False):
    headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36'}
    if "ی" in symbol:
        symbol = symbol.replace("ی","ي")
    id = stock_id(symbol)
    url = 'http://www.tsetmc.com/tsev2/data/Export-txt.aspx?t=i&a=1&b=0&i=' + id
    html=rq.get(url,headers=headers).content
    df=pd.read_csv(io.StringIO(html.decode("utf-8")),index_col="<DTYYYYMMDD>",parse_dates=True)[::-1]
    df=df.rename(columns={"<OPEN>": "Yesterday", "<CLOSE>": "Close","<FIRST>":"Open","<HIGH>":"High","<LOW>":"Low","<VOL>":"Volume"})
    df.index.rename('Date',inplace=True)
    df = df[["Open","High","Low","Close","Volume","Yesterday"]]
    df["Date"] = df.index
    df["JDate"] = df["Date"].jalali.to_jalali()
    if j_date:
        if start != None:
            start = jdatetime.date(int(start[:4]),int(start[5:7]),int(start[8:])).togregorian()
            start = str(start.year)+"-"+str(start.month) + "-" + str(start.day)
        if end != None:
            end = jdatetime.date(int(end[:4]),int(end[5:7]),int(end[8:])).togregorian()
            end = str(end.year)+"-"+str(end.month)+"-"+str(end.day)
    if start != None:
        df = df[start:]
    if end != None:
        df = df[:end]
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
    df.drop(["Yesterday", "Date"], axis=1, inplace=True)
    return df

#----------------------------------------


def pricing_based_on_stock(stock_name:str, trading_days:int=100, IV=False, leverage=True, P_BSM=False, sort="Maturity"):
    """
    Valuation of all options on a particular stock
    """
    if "ی" in stock_name:
        stock_name = stock_name.replace("ی","ي")
    last_price = stock_price(stock_name)
    df_stock = download_history(stock_name, start=None, end=None, adjust_price=True)[-trading_days:]
    df = df_stock[["Adj Close"]].copy()
    df["return"] = np.log(df["Adj Close"]/df["Adj Close"].shift())
    sigma = df["return"].std() * np.sqrt(240)
    rf = risk_free_interest_rate()
    data = tse_options(stock_name)
    if data.empty:
        data = farabourse_options(stock_name)

    col = ["Symbol", "Strike Price", "Status", "Maturity(day)","Open Interest", "Bid", "Ask", "Delta"]
    if IV == True:
        col.append("IV")
    if leverage == True:
        col.append("Leverage")
    col.append("BSM")
    if P_BSM == True:
        col.append("%Price/BSM")
    df = pd.DataFrame(index=range(1,1+len(data)),columns=col)

    for i in range(len(data)):
        df.iloc[i]["Symbol"] = data.iloc[i]["نماد"]
        df.iloc[i]["Stock"] = stock_name
        df.iloc[i]["Strike Price"] = int(data.iloc[i]["قیمت اعمال"])
        if last_price > 1.025*df.iloc[i]["Strike Price"]:
            df.iloc[i]["Status"] = "ITM"
        elif last_price < 0.975*df.iloc[i]["Strike Price"]:
            df.iloc[i]["Status"] = "OTM"
        else:
            df.iloc[i]["Status"] = "ATM"
        df.iloc[i]["Maturity(day)"] = int(data.iloc[i]["روزهای باقیمانده تا سررسید"])
        df.iloc[i]["Open Interest"] = data.iloc[i]["موقیعت های باز"]
        df.iloc[i]["Bid"] = data.iloc[i]["حجم بهترین سفارش خرید"]
        df.iloc[i]["Ask"] = data.iloc[i]["قیمت بهترین سفارش فروش"]
 

    if sort.upper() == "MATURITY":
        df = df.sort_values(by=['Maturity(day)'], ascending=True)
    elif sort.upper() == "STRIKE PRICE":
        df = df.sort_values(by=['Strike Price'], ascending=True)
    else:
        df = df.sort_values(by=['Open Interest'], ascending=False)
    df = df.reset_index(drop=True)

    for i in range(len(df)):
        K = df["Strike Price"].iloc[i]
        T = df["Maturity(day)"].iloc[i]/360
        d1 = (np.log(last_price/K) + (rf + (sigma ** 2 )/2) * T)/(sigma * (T ** 0.5))
        d2 = d1 - (sigma * (T ** 0.5))
        df["BSM"].iloc[i] = round((last_price * norm.cdf(d1)) - (K * np.e ** (-rf*T) * norm.cdf(d2)))
        market_price = int(df["Ask"].iloc[i])
        df["Delta"].iloc[i] = round(norm.cdf(d1),2)
        if IV == True:
            if market_price == 0:
                df["IV"].iloc[i] = "%-"
            else:
                df["IV"].iloc[i] = f"%{round(find_IV(market_price, last_price, K, rf, T)*100,1)}"
                if df["IV"].iloc[i]=='%nan':
                    df["IV"].iloc[i] = "%-"
            

        if leverage == True:    
            try:
                df["Leverage"].iloc[i] = round(df["Delta"].iloc[i] * last_price / df["Ask"].iloc[i], 2)
            except:
                df["Leverage"].iloc[i] = np.inf
        if P_BSM == True:
            try:
                price_bsm = round(-100*(market_price/df.iloc[i]["BSM"]-1),1)
                
                if price_bsm >=0:
                    df["%Price/BSM"].iloc[i] = f"%{price_bsm}\U0001f7e2"
                else:
                    df["%Price/BSM"].iloc[i] = f"%{price_bsm}\U0001F534"
                if market_price == 0:
                    df["%Price/BSM"].iloc[i] = "%-"
            except:
                df["%Price/BSM"].iloc[i] = "%-"


    df.index = df["Symbol"]
    df = df.drop("Symbol", axis=1)
    print(f"Stock Price: {int(last_price)} \tRiskFreeRate: {round(rf*100,2)}% \tHV: {round(sigma*100,2)}%")
    return df

#----------------------------------------

def pricing_based_on_option(option_name:str, trading_days:int=100, IV=False, leverage=True, P_BSM=False):
    """
    Valuation of a particular option
    """
    if "ی" in option_name:
        option_name = option_name.replace("ی","ي")
    headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36'}
    url = f"http://www.tsetmc.com/tsev2/data/search.aspx?skey={option_name}"
    html = rq.get(url, headers=headers).text
    if len(html.split(";")) == 2:
        stock_name = html.split(",")[1].split("-")[0].split(" ")[1]
        last_price = stock_price(stock_name)
        df_stock = download_history(stock_name, start=None, end=None, adjust_price=True)[-trading_days:]
        df = df_stock[["Adj Close"]].copy()
        df["return"] = np.log(df["Adj Close"]/df["Adj Close"].shift())
        sigma = df["return"].std() * np.sqrt(240)
        rf = risk_free_interest_rate()
        option_in_tse = option_name + "1"
        data = tse_options(option_in_tse, False)
        if data.empty:
            data = farabourse_options(option_name, False)
        
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
            df = pd.DataFrame(index=range(1,1+len(data)),columns=col)

            df.iloc[0]["Symbol"] = data.iloc[0]["نماد"]
            df.iloc[0]["Stock"] = stock_name
            df.iloc[0]["Strike Price"] = data.iloc[0]["قیمت اعمال"]
            if last_price > 1.025*df.iloc[0]["Strike Price"]:
                df.iloc[0]["Status"] = "ITM"
            elif last_price < 0.975*df.iloc[0]["Strike Price"]:
                df.iloc[0]["Status"] = "OTM"
            else:
                df.iloc[0]["Status"] = "ATM"
            df.iloc[0]["Maturity(day)"] = int(data.iloc[0]["روزهای باقیمانده تا سررسید"])
            df.iloc[0]["Open Interest"] = data.iloc[0]["موقیعت های باز"]
            df.iloc[0]["Bid"] = data.iloc[0]["حجم بهترین سفارش خرید"]
            df.iloc[0]["Ask"] = data.iloc[0]["قیمت بهترین سفارش فروش"]

            df = df.reset_index(drop=True)

            K = df["Strike Price"].iloc[0]
            T = df["Maturity(day)"].iloc[0]/360
            d1 = (np.log(last_price/K) + (rf + (sigma ** 2 )/2) * T)/(sigma * (T ** 0.5))
            d2 = d1 - (sigma * (T ** 0.5))
            df["BSM"].iloc[0] = round((last_price * norm.cdf(d1)) - (K * np.e ** (-rf*T) * norm.cdf(d2)))
            market_price = int(df["Ask"].iloc[0])
            df["Delta"].iloc[0] = round(norm.cdf(d1),2)
            if IV == True:
                if market_price == 0:
                    df["IV"].iloc[0] = "%-"
                else:
                    df["IV"].iloc[0] = f"%{round(find_IV(market_price, last_price, K, rf, T)*100,1)}"
                    if df["IV"].iloc[0]=='%nan':
                        df["IV"].iloc[0] = "%-"
            if leverage == True:
                try:
                    df["Leverage"].iloc[0] = round(df["Delta"].iloc[0] * last_price /market_price, 2)
                except:
                    df["Leverage"].iloc[0] = np.inf

            if P_BSM == True:
                try:
                    price_bsm = round(-100*(market_price/df.iloc[0]["BSM"]-1),1)
                    if price_bsm >=0:
                        df["%Price/BSM"].iloc[0] = f"%{price_bsm}\U0001f7e2"
                    else:
                        df["%Price/BSM"].iloc[0] = f"%{price_bsm}\U0001F534"
                    if market_price == 0:
                        df["%Price/BSM"].iloc[0] = "%-"
                except:
                    df["%Price/BSM"].iloc[0] = "%-"

            df.index = df["Symbol"]
            df = df.drop("Symbol", axis=1)
            print(f"Stock Price: {int(last_price)}\tRiskFreeRate: {round(rf*100,2)}%\tHV: {round(sigma*100,2)}%")
            return df


   


    elif len(html.split(";")) == 1:
        raise NameError("This symbol does not exist. Please enter the symbol correctly!")
    else:
        raise NameError("Please enter the symbol completely!")
















"""""
def pricing_based_on_option(option_name:str, trading_days:int=100, IV=False):

    headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36'}
    url = f"http://www.tsetmc.com/tsev2/data/search.aspx?skey={option_name}"
    html = rq.get(url, headers=headers).text
    if len(html.split(";")) == 2:
        stock_name = html.split(",")[1].split("-")[0].split(" ")[1]
        df = pricing_based_on_stock(stock_name, trading_days, IV)
        try:
            df = pd.DataFrame(df.loc[f"{option_name}1"])
        except KeyError:
            try:
                df = pd.DataFrame(df.loc[option_name])
            except KeyError:
                print("This symbol does not exist. Please enter the symbol correctly!")

        return df
    elif len(html.split(";")) == 1:
        print("This symbol does not exist. Please enter the symbol correctly!")
    else:
        print("Please enter the symbol completely!")


"""