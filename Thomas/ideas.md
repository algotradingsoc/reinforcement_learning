# Ideas 

## Feature construction 
Given data for log-return, log-turnover, implied volatility, generate tsfresh features
(rank of recent entry, mean, std, skew, kurtosis, median, min, max, etc) for these metrics of various lookback period (21,63,252)

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

Reward function can be volatility scale log-returns for each time period 


## Gym environment 
Randomly start and end backtest in the training period (to encourage robustness?)

Use the DataLoader to save csv files for features. 

Define t-cost as a sum of bid-ask spread and fixed cost 

Lookback period: 25, 90, 250. How does lookback period affect trading strategies 

Trade frequency: To reduce t-cost, trade only on Friday

Reward function: log_return / volatility, needs to normalised reward by volatility for better training 





