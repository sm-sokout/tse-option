# tse_option


English | [فارسی](README.fa.md)


`tse_option` is a Python package for retrieving and analyzing options
data from the Tehran Stock Exchange (TSE) and Iran Fara Bourse (IFB).

Parts of this project build on functions adapted from
[`finpy_tse`](https://github.com/ARahimiQuant/finpy-tse) and
[`tsemodule5`](https://github.com/python4financeacademy/tsemodule5).


-   Telegram channel: [@algorithm_edge](https://t.me/algorithm_edge)

------------------------------------------------------------------------

> **Disclaimer:** This package, including its pricing and
> implied-volatility calculations, is provided for analytical and
> decision-support purposes only. Nothing produced by the package should
> be construed as investment advice or a recommendation to buy or sell
> any security or derivative. Users are solely responsible for their
> investment decisions and any resulting gains or losses. The developer
> accepts no liability for losses arising from the use of this package.

------------------------------------------------------------------------

## Release Notes

### Version 0.1.1.0

1.  Added support for downloading historical price data for stocks and
    option contracts.
2.  Fixed several issues.

### Version 0.1.2.1

1.  Updated TSETMC links.
2.  Added support for downloading historical price data for multiple
    symbols in a single request, similar to `yfinance`.
3.  Updated links to `tse.ir`.

### Version 0.1.2.3

1.  Fixed the risk-free rate calculation based on the average rate of
    Iranian Treasury bills (Akhza).
2.  General improvements and bug fixes.

### Version 0.1.3.0

1.  Added support for retrieving put-option data from the Tehran Stock
    Exchange.
2.  Fixed issues with retrieving data from Iran Fara Bourse.
3.  Added a margin requirement column.

### Version 0.1.4.0

1.  Fixed an issue that prevented options data from being retrieved.
2.  Added open interest data for each option contract.
3.  Added a manual fallback for the risk-free rate when automatic
    calculation fails.

------------------------------------------------------------------------

### Upgrade

``` bash
pip install tse-option --upgrade
```

### Installation

``` bash
pip install tse-option
```

### Import

``` python
import tse_option as tso
```

------------------------------------------------------------------------

#### Retrieve the Option Chain for an Underlying Asset

``` python
df = tso.option_chain(symbol="خودرو", trading_days=100, IV=False, leverage=True, P_BSM=False, sort="Maturity")
```

  -----------------------------------------------------------------------
  Argument                       Description
  ------------------------------ ----------------------------------------
  `symbol`                       Underlying asset symbol

  `trading_days`                 Number of trading days used to calculate
                                 historical volatility

  `IV`                           Whether to calculate implied volatility

  `leverage`                     Whether to calculate leverage

  `P_BSM`                        Whether to calculate the ratio of the
                                 market price to the Black-Scholes-Merton
                                 (BSM) price

  `sort`                         Field used to sort the results
  -----------------------------------------------------------------------

The results can be sorted by fields such as time to maturity
(`Maturity`), strike price (`Strike Price`), or open interest
(`Open Interests`).

------------------------------------------------------------------------

#### Retrieve Call Option Data

``` python
df = tso.call(option_symbol="ضخود1130", trading_days=100, IV=False, leverage=True, P_BSM=False)
```

  -----------------------------------------------------------------------
  Argument                       Description
  ------------------------------ ----------------------------------------
  `option_symbol`                Call option symbol

  `trading_days`                 Number of trading days used to calculate
                                 historical volatility

  `IV`                           Whether to calculate implied volatility

  `leverage`                     Whether to calculate leverage

  `P_BSM`                        Whether to calculate the ratio of the
                                 market price to the Black-Scholes-Merton
                                 (BSM) price
  -----------------------------------------------------------------------

------------------------------------------------------------------------

#### Retrieve Put Option Data

``` python
df = tso.put(option_symbol="طخود1138", trading_days=100, IV=False, leverage=True, P_BSM=False)
```

  -----------------------------------------------------------------------
  Argument                       Description
  ------------------------------ ----------------------------------------
  `option_symbol`                Put option symbol

  `trading_days`                 Number of trading days used to calculate
                                 historical volatility

  `IV`                           Whether to calculate implied volatility

  `leverage`                     Whether to calculate leverage

  `P_BSM`                        Whether to calculate the ratio of the
                                 market price to the Black-Scholes-Merton
                                 (BSM) price
  -----------------------------------------------------------------------

------------------------------------------------------------------------

#### Download Historical Price Data

For a single symbol:

``` python
df = tso.download("خودرو", j_date=True, start="1402-01-01", end=None, adjust_price=True, drop_unadjusted=False)
```

For multiple symbols:

``` python
df = tso.download(symbols=["خودرو","فولاد","وبملت"], j_date=False, start="2023-01-01", end=None, adjust_price=False, drop_unadjusted=False)
```

  -----------------------------------------------------------------------
  Argument                       Description
  ------------------------------ ----------------------------------------
  `symbols`                      A symbol or a list of symbols

  `j_date`                       Whether to use Jalali dates

  `start`                        Start date

  `end`                          End date

  `adjust_price`                 Whether to adjust historical prices

  `drop_unadjusted`              Whether to remove unadjusted prices from
                                 the output
  -----------------------------------------------------------------------

------------------------------------------------------------------------

For additional examples, see the [example
notebook](https://github.com/sm-sokout/tse-option/blob/master/Example/Example.ipynb).

------------------------------------------------------------------------

**Telegram:** [@algorithm_edge](https://t.me/algorithm_edge)

If you encounter a bug or any unexpected behavior, please feel free to
report it at `sm.sokut@gmail.com`.

**GitHub:** [tse-option](https://github.com/sm-sokout/tse-option)
