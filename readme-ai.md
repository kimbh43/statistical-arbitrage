## Overview

This repository contains an implementation of a statistical arbitrage strategy known as Pair Trading. Pair trading is a market-neutral trading strategy that capitalizes on the historical price relationships between two stocks. When the price spread between the two correlated stocks diverges, a pair trade is executed with the expectation that the spread will revert to its mean, thereby creating a profit opportunity.

## Features

- **Data Collection**: Scripts to facilitate the downloading of historical prices for stock pairs. yahoo finance.
- **Cointegration Testing**: Functions to assess whether a pair of stocks exhibit a stable, long-term relationship suitable for pair trading. Augmented Dicky Fuller Test and Johansen Test were performed on pairs selected via normalized price distance.
- **Signal Generation**: Algorithms to pinpoint optimal entry and exit points for trades based on the divergence of the stock pair spread. Least squares, rolling least squares and Kalman filter was used to generate z-score. Signal was constructed based off relationship between z-score and long/short threshold.
- **Backtesting**: A backtesting engine to simulate pair trading strategies on historical data, allowing for performance evaluation. Backtrader was used for backtesting.

---

## Repository Structure

```sh
└── statistical-arbitrage/
    ├── LICENSE
    ├── README.md
    ├── data
    │   ├── AVB.csv
    │   ├── CMS.csv
    │   ├── CNP.csv
    │   ├── DTE.csv
    │   ├── INVH.csv
    │   ├── LNT.csv
    │   ├── PPL.csv
    │   ├── SRE.csv
    │   ├── sp500_daily.csv
    │   └── spx.csv
    ├── docs
    │   ├── Makefile
    │   ├── conf.py
    │   └── make.bat
    ├── requirements.txt
    ├── setup.py
    ├── src
    │   ├── .DS_Store
    │   ├── dataframe_result.xlsx
    │   ├── pair_trading_backtest.py
    │   ├── pair_trading_sp500.ipynb
    │   ├── pyfolio_positions_tear_sheet.png
    │   ├── pyfolio_returns_tear_sheet.png
    │   └── quant
    │       ├── __init__.py
    │       ├── arbitrage.py
    │       ├── helpers.py
    │       └── screening.py
    └── tests
        ├── __init__.py
        ├── context.py
        └── test_basic.py
```

---

## Modules

| File                                                                                                  | Summary                              |
| ----------------------------------------------------------------------------------------------------- | ------------------------------------ |
| [requirements.txt](https://github.com/kimbh43/statistical-arbitrage.git/blob/master/requirements.txt) | <code>► requirements.txt file</code> |
| [setup.py](https://github.com/kimbh43/statistical-arbitrage.git/blob/master/setup.py)                 | <code>► setup.py file</code>         |

</details>

| File                                                                                                                      | Summary                                                               |
| ------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------- |
| [pair_trading_sp500.ipynb](https://github.com/kimbh43/statistical-arbitrage.git/blob/master/src/pair_trading_sp500.ipynb) | <code>► Report on pair trading strategy and backtesting result</code> |
| [pair_trading_backtest.py](https://github.com/kimbh43/statistical-arbitrage.git/blob/master/src/pair_trading_backtest.py) | <code>► Backtesting wrote using backtrader </code>                    |

</details>

| File                                                                                                    | Summary                                                                                      |
| ------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------- |
| [arbitrage.py](https://github.com/kimbh43/statistical-arbitrage.git/blob/master/src/quant/arbitrage.py) | <code>► Functions regarding statistical cointegration testing and pair selection</code>      |
| [screening.py](https://github.com/kimbh43/statistical-arbitrage.git/blob/master/src/quant/screening.py) | <code>► Functions that generates gamma and mu value for z-score and signal generation</code> |
| [helpers.py](https://github.com/kimbh43/statistical-arbitrage.git/blob/master/src/quant/helpers.py)     | <code>► Helper functions</code>                                                              |

</details>

---

## Getting Started

**_Requirements_**

Ensure you have the following dependencies installed on your system:

- **Python**: `version 2.7.18`

### Installation

1. Clone the statistical-arbitrage repository:

```sh
git clone https://github.com/kimbh43/statistical-arbitrage.git
```

2. Change to the project directory:

```sh
cd statistical-arbitrage
```

3. Install the dependencies:

```sh
pip install -r requirements.txt
```

### Running `statistical-arbitrage`

Use the following command to run backtesing statistical-arbitrage:

```sh
python3 src/pair_trading_backtest.py
```

---

<details closed>
    <summary>Contributing Guidelines</summary>

1. **Fork the Repository**: Start by forking the project repository to your github account.
2. **Clone Locally**: Clone the forked repository to your local machine using a git client.
   ```sh
   git clone https://github.com/kimbh43/statistical-arbitrage.git
   ```
3. **Create a New Branch**: Always work on a new branch, giving it a descriptive name.
   ```sh
   git checkout -b new-feature-x
   ```
4. **Make Your Changes**: Develop and test your changes locally.
5. **Commit Your Changes**: Commit with a clear message describing your updates.
   ```sh
   git commit -m 'Implemented new feature x.'
   ```
6. **Push to GitHub**: Push the changes to your forked repository.
   ```sh
   git push origin new-feature-x
   ```
7. **Submit a Pull Request**: Create a PR against the original project repository. Clearly describe the changes and their motivations.

Once your PR is reviewed and approved, it will be merged into the main branch.

</details>

---

## Acknowledgments

- [1] Gatev, E., Goetzmann, W. N., & Rouwenhorst, K. G. (2006). Pairs trading: Performance of a relative-value arbitrage rule. The Review of Financial Studies, 19(3), 797-827.
- [2] Vidyamurthy, G. (2004). Pairs Trading: quantitative methods and analysis (Vol. 217). John Wiley & Sons.

---
