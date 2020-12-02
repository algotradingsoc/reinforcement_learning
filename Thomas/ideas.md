# Ideas 

## SAC and PPO 
Study the parameter settings of SAC and PPO on ray, in particular the replay buffer and training batch size. Does these parameters affect training?

How to create a user defined model in ray for SAC policy. The default SACmodel is a good starting point.
A model would consist of CNN, LSTM and dense layers. 

Standardised input features before passing to the CNN (can be done in get_observation function)

## Reward function 
Reward function is return penalised by drawdown or return penalised by volatility
Challenge: Sharpe ratio is not well-defined during the first 60 trading days. 
Cumulative return depends on length of backtest, how about use annualised versions 

Use the sharpe ratio or the difference of the sharpe ratio

Randomly start and end backtest in the training period (to encourage robustness?)


## Universe and features

Recreate typical trading strategies using RL 

Universe 1: ETF rotations   
SPY, QQQ, EEM, TLT, GLD, LQD

Single stock universe: TSLA 

Big stocks universe: BRK, GE, GOLD, AAPL, GS, T 

Use the DataLoader to save csv files for features. We will use the return, t_cost, volatility_20, skewness_20 and kurtosis_20 

Define t-cost as a sum of bid-ask spread and fixed cost 

Lookback period: 25, 90, 250. How does lookback period affect trading strategies 

Trade frequency: To reduce t-cost, trade only on Friday

Reward function: log_return / volatility, needs to normalised reward by volatility for better training 





