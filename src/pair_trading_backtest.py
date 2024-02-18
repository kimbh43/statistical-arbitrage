import os
import pandas as pd
from datetime import datetime
import backtrader as bt
import quant.arbitrage
import quant.helpers
import matplotlib
matplotlib.use('Agg')

class PairTradingStrategy(bt.Strategy):
    params = (
        ('pairs', None),  # A list of tuples, each tuple containing the symbols of a pair
        ('z_score', None),
        ('spread', None),
        ('weight', None),
        ('target_ratio', None),
    )

    def __init__(self):
        self.buyprice = {}
        self.buycomm = {}
        self.bar_executed = {}
        self.val_start = None
        self.pairs = self.params.pairs
        self.orders = {pair: None for pair in self.pairs}
        self.threshold_short = 1
        self.threshold_long = -1
        self.pairs_data = {pair: (self.getdatabyname(pair[0]), self.getdatabyname(pair[1])) for pair in self.pairs}
        self.pairs_position = {pair: 0 for pair in self.pairs}
        self.zscore = self.params.z_score
        self.spread = self.params.spread
        self.weight = self.params.weight
        self.counter = 0
        self.utilized = self.params.target_ratio
    
    def next(self):
        if any(self.orders.values()):
            return
        for pair in self.pairs:
            current_z_score = self.zscore[pair].iloc[self.counter]
            current_spread = self.spread[pair].iloc[self.counter]
            current_w1, current_w2 = self.weight[pair][self.counter]
            total_value = self.broker.getvalue()  
            invest_amount = (total_value / len(self.pairs) * self.utilized)  

            
            symbol_1_data, symbol_2_data = self.pairs_data[pair]

            # Calculate allocation for each stock in the pair
            allocation_stock_1 = invest_amount * current_w1 / (current_w1 + current_w2 * -1)
            allocation_stock_2 = invest_amount * current_w2 * -1 / (current_w1 + current_w2 * -1)

            # Calculate size for each stock in the pair
            size_stock_1 = allocation_stock_1 / symbol_1_data.close[0]
            size_stock_2 = allocation_stock_2 / symbol_2_data.close[0]

            # self.log(f'{current_z_score},{symbol_1_data.close[0]}, {symbol_2_data.close[0]}, {self.position}, {current_w1}, {current_w2}', pair = pair)

            if not self.pairs_position[pair]:
                if current_z_score <= self.threshold_long:
                    self.orders[pair] = (
                        self.buy(data=symbol_1_data, size=size_stock_1),
                        self.sell(data=symbol_2_data, size=size_stock_2)
                    )
                    self.pairs_position[pair] += size_stock_1
                elif current_z_score >= self.threshold_short:
                    self.orders[pair] = (
                        self.sell(data=symbol_1_data, size=size_stock_1),
                        self.buy(data=symbol_2_data, size=size_stock_2)
                    )
                    self.pairs_position[pair] += size_stock_2
            elif (self.pairs_position[pair] > 0 and current_z_score <= 0) or (self.pairs_position[pair] < 0 and current_z_score >= 0):
                self.orders[pair] = (
                    self.close(data=symbol_1_data),
                    self.close(data=symbol_2_data)
                )
                self.pairs_position[pair] = 0


        self.counter += 1

    def log(self, text, dt=None, doprint=False, pair=None):
        ''' Logging function for this strategy'''
        dt = dt or self.datas[0].datetime.date(0)
        pair_text = f'Pair {pair}: ' if pair else ''
        print(f'{dt.isoformat()}, {pair_text}{text}')  # Print date and close


    def start(self):
        self.val_start = self.broker.get_cash()  # keep the starting cash

    def notify_trade(self, trade):
        if not trade.isclosed:
            return
        self.log('OPERATION PROFIT, GROSS %.2f, NET %.2f' % (trade.pnl, trade.pnlcomm))

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            # Buy/Sell order submitted/accepted to/by broker - Nothing to do
            return

        # Check if an order has been completed
        # Attention: broker could reject order if not enough cash
        if order.status in [order.Completed]:
            pair = None
            for p, orders in self.orders.items():
                if orders and order in orders:
                    pair = p
                    break

            if pair is not None:
                if order.isbuy():
                    self.log(
                        'BUY EXECUTED, Price: %.2f, Size: %.0f, Cost: %.2f, Comm %.2f, RemSize: %.0f, RemCash: %.2f' %
                        (order.executed.price,
                         order.executed.size,
                         order.executed.value,
                         order.executed.comm,
                         order.executed.remsize,
                         self.broker.get_cash()))
    
                    self.buyprice = order.executed.price
                    self.buycomm = order.executed.comm

                else:  # Sell
                    self.log('SELL EXECUTED, Price: %.2f, Size: %.0f, Cost: %.2f, Comm %.2f, RemSize: %.0f, RemCash: %.2f' %
                             (order.executed.price,
                              order.executed.size,
                              order.executed.value,
                              order.executed.comm,
                              order.executed.remsize,
                              self.broker.get_cash()))
                self.orders[pair] = None
            self.bar_executed = len(self)
        elif order.status in [order.Canceled, order.Expired, order.Margin, order.Rejected]:
            self.log('Order Failed', pair=pair)
            # Remove the failed order from the orders dictionary
            if pair is not None:
                self.orders[pair] = None

    def stop(self):
        # calculate the actual returns
        print(self.analyzers)
        roi = (self.broker.get_value() / self.val_start) - 1.0
        self.log('ROI:        {:.2f}%'.format(100.0 * roi))

if __name__ == '__main__':
    param_opt = False
    perf_eval = True
    benchmark = 'SPX'

    cerebro = bt.Cerebro()

    tickers = [
    'DTE',
    'CMS',
    'AVB',
    'INVH',
    'CNP',
    'SRE',
    'LNT',
    'PPL'
    ]
    
    # Loop through the tickers and add a data feed for each one
    for ticker in tickers:
        data = bt.feeds.YahooFinanceCSVData(
            dataname=f'data/{ticker}.csv',
            # Specify your fromdate and todate if needed
            fromdate=datetime(2023, 8, 1),
            todate=datetime(2024, 1, 31),
            reverse=False
        )
        cerebro.adddata(data,name=ticker)

    # 1	AVB	INVH	0.352817	Real Estate	Real Estate o
    # 2	CNP	SRE	0.442587	Utilities	Utilities o
    # 3	DTE	LNT	0.511368	Utilities	Utilities o
    # 4	CMS	DTE	0.526707	Utilities	Utilities o
    # 5	PPL	SRE	0.540921	Utilities	Utilities o
    sp500_prices_df = pd.read_csv('data/sp500_daily.csv', index_col='Date')
    filtered_data = quant.helpers.filter_data(sp500_prices_df)

    gamma_1 = quant.arbitrage.estimate_mu_gamma_rolling_LS(filtered_data["CMS"],filtered_data["DTE"], 100, 10, 1)['gamma']
    mu_1 = quant.arbitrage.estimate_mu_gamma_rolling_LS(filtered_data["CMS"],filtered_data["DTE"], 100, 10, 1)['mu']
    weight_1, spread_1 = quant.arbitrage.compute_spread(filtered_data["CMS"],filtered_data["DTE"],gamma_1, mu_1)
    z_score_1, signal_1 = quant.arbitrage.generate_signal(spread_1, -1, 1, pct_training = 2/3)

    gamma_2 = quant.arbitrage.estimate_mu_gamma_rolling_LS(filtered_data["AVB"],filtered_data["INVH"], 100, 10, 1)['gamma']
    mu_2 = quant.arbitrage.estimate_mu_gamma_rolling_LS(filtered_data["AVB"],filtered_data["INVH"], 100, 10, 1)['mu']
    weight_2, spread_2 = quant.arbitrage.compute_spread(filtered_data["AVB"],filtered_data["INVH"],gamma_2,mu_2)
    z_score_2, signal_2 = quant.arbitrage.generate_signal(spread_2, -1, 1, pct_training = 2/3)

    
    gamma_3 = quant.arbitrage.estimate_mu_gamma_rolling_LS(filtered_data["CNP"],filtered_data["SRE"], 100, 10, 1)['gamma']
    mu_3 = quant.arbitrage.estimate_mu_gamma_rolling_LS(filtered_data["CNP"],filtered_data["SRE"], 100, 10, 1)['mu']
    weight_3, spread_3 = quant.arbitrage.compute_spread(filtered_data["CNP"],filtered_data["SRE"],gamma_3,mu_3)
    z_score_3, signal_3 = quant.arbitrage.generate_signal(spread_3, -1, 1, pct_training = 2/3)

        
    gamma_4 = quant.arbitrage.estimate_mu_gamma_rolling_LS(filtered_data["DTE"],filtered_data["LNT"], 100, 10, 1)['gamma']
    mu_4 = quant.arbitrage.estimate_mu_gamma_rolling_LS(filtered_data["DTE"],filtered_data["LNT"], 100, 10, 1)['mu']
    weight_4, spread_4 = quant.arbitrage.compute_spread(filtered_data["DTE"],filtered_data["LNT"],gamma_4,mu_4)
    z_score_4, signal_4 = quant.arbitrage.generate_signal(spread_4, -1, 1, pct_training = 2/3)

    gamma_5 = quant.arbitrage.estimate_mu_gamma_rolling_LS(filtered_data["PPL"],filtered_data["SRE"], 100, 10, 1)['gamma']
    mu_5 = quant.arbitrage.estimate_mu_gamma_rolling_LS(filtered_data["PPL"],filtered_data["SRE"], 100, 10, 1)['mu']
    weight_5, spread_5 = quant.arbitrage.compute_spread(filtered_data["PPL"],filtered_data["SRE"],gamma_5,mu_5)
    z_score_5, signal_5 = quant.arbitrage.generate_signal(spread_5, -1, 1, pct_training = 2/3)

    cerebro.addstrategy(PairTradingStrategy, 
                        pairs = [("CMS", "DTE"),("AVB", "INVH"),("CNP", "SRE"),("DTE", "LNT"),("PPL", "SRE")], 
                        z_score = {("CMS", "DTE"): z_score_1,("AVB", "INVH"): z_score_2,("CNP", "SRE"): z_score_3,("DTE", "LNT"): z_score_4,("PPL", "SRE"):z_score_5},
                        spread = {("CMS", "DTE"): spread_1,("AVB", "INVH"): spread_2, ("CNP", "SRE"): spread_3, ("DTE", "LNT"): spread_4, ("PPL", "SRE"): spread_5},
                        weight = {("CMS", "DTE"): weight_1,("AVB", "INVH"): weight_2, ("CNP", "SRE"): weight_3, ("DTE", "LNT"): weight_4,("PPL", "SRE"): weight_5},
                        target_ratio = 0.95) 
    # Set our desired cash start
    cerebro.broker.setcash(2000000.0)
    # Set the commission - 0.01% ... divide by 100 to remove the %
    cerebro.broker.setcommission(commission=0.0001)

    # Print out the starting conditions
    print('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())
    
    # Add Analyzer
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='SharpeRatio')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='DrawDown')
    cerebro.addanalyzer(bt.analyzers.PyFolio, _name='pyfolio')
    results = cerebro.run()

    # Print out the final result
    strat = results[0]
    print('Final Portfolio Value: %.2f, Sharpe Ratio: %.2f, DrawDown: %.2f, MoneyDown %.2f' %
          (cerebro.broker.getvalue(),
           strat.analyzers.SharpeRatio.get_analysis()['sharperatio'],
           strat.analyzers.DrawDown.get_analysis()['drawdown'],
           strat.analyzers.DrawDown.get_analysis()['moneydown']))

    if perf_eval:
        import matplotlib.pyplot as plt
        cerebro.plot(style='candlestick')
        plt.show()

        pyfoliozer = strat.analyzers.getbyname('pyfolio')
        returns, positions, transactions, gross_lev = pyfoliozer.get_pf_items()
        print('-------------- RETURNS ----------------')
        print(returns)
        print('-------------- POSITIONS ----------------')
        print(positions)
        print('-------------- TRANSACTIONS ----------------')
        print(transactions)
        print('-------------- GROSS LEVERAGE ----------------')
        print(gross_lev)

        import empyrical as ep
        import pyfolio as pf

        benchmark = "SPX"

        bm_ret = None
        if benchmark:
            datapath = os.path.join('./data/', f'{benchmark}.csv')
            bm = pd.read_csv(datapath, index_col=0)
            bm_ret = bm['Close'].pct_change().dropna()
            bm_ret.index = pd.to_datetime(bm_ret.index)
            returns.index = returns.index.tz_localize(None)
            bm_ret = bm_ret[returns.index]
            bm_ret.name = 'benchmark'

        perf_stats_strat = pf.timeseries.perf_stats(returns)
        perf_stats_all = perf_stats_strat
        if benchmark:
            perf_stats_bm = pf.timeseries.perf_stats(bm_ret)
            perf_stats_all = pd.concat([perf_stats_strat, perf_stats_bm], axis=1)
            perf_stats_all.columns = ['Strategy', 'Benchmark']
        print(returns)

        drawdown_table = pf.timeseries.gen_drawdown_table(returns, 5)
        monthly_ret_table = ep.aggregate_returns(returns, 'monthly')
        monthly_ret_table = monthly_ret_table.unstack().round(3)
        ann_ret_df = pd.DataFrame(ep.aggregate_returns(returns, 'yearly'))
        ann_ret_df = ann_ret_df.unstack().round(3)
       
        print('-------------- PERFORMANCE ----------------')
        print(perf_stats_all)
        print('-------------- DRAWDOWN ----------------')
        print(drawdown_table)
        print('-------------- MONTHLY RETURN ----------------')
        print(monthly_ret_table)
        print('-------------- ANNUAL RETURN ----------------')
        print(ann_ret_df)

        # f = pf.create_position_tear_sheet(
        #     returns,
        #     # benchmark_rets=bm_ret if benchmark else None,
        #     positions=positions,
        #     transactions=transactions,
        #     # round_trips=False,
        #     return_fig=True     
        #     )
        f = pf.create_returns_tear_sheet(
            returns,
            benchmark_rets=bm_ret if benchmark else None,
            positions=positions,
            transactions=transactions,
            # round_trips=False,
            return_fig=True     
            )
        f.savefig('./pyfolio_returns_tear_sheet.png')