# Reinforcement Learning 


## Overview 

Using deep reinforcement learning to generate trading signals for a basket of related US Equities using price data and financial ratios.
We will create an OpenAI gym environment to simulate the daily rebalancing of US Equities 
The project will start with using standard policies and model designs from ray, a scalable framework for reinforcement learning. 
We will explore different ways of feature engineering, reward function, neural-network designs.
As the flagship project in research, results from the generative models and price forecast teams will be integrated into the RL model.


## Resources and skills invovled 

Python: Tensorflow/PyTorch, Ray, OpenAI gym, 
Database: MongoDB 


## Data Sources 

Kaggle US Equities Data (1992-2019)
Kaggle US reported financial Data (2010-2020)

https://www.kaggle.com/finnhub/reported-financials
https://www.kaggle.com/finnhub/end-of-day-us-market-data-survivorship-bias-free

The price data is survivorship bias free with dividend adjustments 


## Project stages 

Stage 1: Build a gym environment to simulate rebalancing (daily/minute) for US Equities 

Design input space [Timestamp * features * n_assets] (2D or 3D?)
Design action space [n_assets + cash] 
Design rebalance logic 
Find a suitable transaction cost model (fixed cost or depend on volatility of asset) 
Work with the data infrastrcture team to generate the required features for the input space
Work with the price forecast team to use some of the price forecasts as a signal for the input space

Details   
The output from RL models is considered to be asset weights for rebalancing.

The input space is log change of price and volume. 
At day T, the value stored is the log price[T] - log price[T-1] 

Rebalance logic 	
At the end of day T, we compute the target asset weights using OHLCV data up to day T,
the log return used for rebalancing is the log return of open price from day T+1 to day T+2. 


Stage 2: Build and train RL models 

Use unsupervised time-series clustering to select the universe of stocks for trading
Explore different hyper-parameters for NN models and architecture using evolutionary strategies and genetic algorithms
Build a suitable RL policy using RLlib and Tensorflow
Train models on our local machine
Build metrics to analyse the robustness of trading signals
Work with the generative models team to use simulated price data to improve robustness of models


Stage 3: Integration

We will build Alphamodels which the asset weights outputs can be used in the Algosoc backtester. 
We will build scripts to automate the training and update of models on AWS sageworker.


### Literature review 

RL policies for continuous action space  

Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor
https://arxiv.org/pdf/1801.01290.pdf  
Proximal Policy Optimization Algorithms  
https://arxiv.org/pdf/1707.06347.pdf  

Applications of RL in trading  

Deep Reinforcement Learning for Trading  
https://arxiv.org/pdf/1911.10107.pdf  
An Application of Deep Reinforcement Learning to Algorithmic Trading		
https://arxiv.org/pdf/2004.06627.pdf


### 






