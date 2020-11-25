import gym
from gym import spaces
from gym.utils import seeding
from empyrical import max_drawdown, alpha_beta, sharpe_ratio, annual_return
import numpy as np
import pandas as pd
from tqdm import tqdm
import sys


#read in the number of lookback days as an argument
Lookback = int(sys.argv[1])


#get ray with alternative temp directory to avoid hpc permission issues
import ray
# Start up Ray. This must be done before we instantiate any RL agents.
ray.init(num_cpus=10, ignore_reinit_error=True, log_to_driver=False,_temp_dir="/rds/general/user/asm119/ephemeral")


#Steal Thomas' data loader and environment

def load_data(price_source='csvdata',tickers=['EEM','QQQ'],start='2008-01-02',end='2010-01-02'):
    '''Returned price data to use in gym environment'''
    # Load data
    # Each dataframe will have columns date and a collection of fields
    # TODO: DataLoader from mongoDB
    # Raw price from DB, forward impute on the trading days for missing date
    # calculate the features (log return, volatility)
    if price_source in ['csvdata']:
        feature_df = []
        price_tensor = []
        for t in tickers:
            df1 = pd.read_csv('/rds/general/user/asm119/home/reinforcement_learning/Thomas/csvdata/{}.csv'.format(t)).set_index('date').loc[start:end]
            feature_df.append(df1)
            price_tensor.append(df1['return']) # assumed to the be log return of the ref price
            ref_df_columns = df1.columns

    # assume all the price_df are aligned and cleaned in the DataLoader
    merged_df = pd.concat(feature_df, axis=1, join='outer')
    price_tensor = np.vstack(price_tensor).transpose()

    return {'dates':merged_df.index, 'fields':ref_df_columns, 'pricedata':price_tensor, 'data':merged_df.values }


class Equitydaily(gym.Env):

    def __init__(self,env_config):

        self.tickers = env_config['tickers']
        self.lookback = env_config['lookback']
        # Load price data, to be replaced by DataLoader class
        raw_data = load_data(env_config['pricing_source'],env_config['tickers'],env_config['start'],env_config['end'])
        # Set the trading dates, features and price data
        self.dates = raw_data['dates']
        self.fields = raw_data['fields']
        self.pricedata = raw_data['pricedata']
        self.featuredata = raw_data['data']
        # Set up historical actions and rewards
        self.n_assets = len(self.tickers) + 1
        self.n_metrics = 2
        self.n_assets_fields = len(self.fields)
        self.n_features = self.n_assets_fields * len(self.tickers) + self.n_assets + self.n_metrics # reward function

        # Set up action and observation space
        # The last asset is cash
        self.action_space = spaces.Box(low=-1, high=1, shape=(len(self.tickers)+1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf,
                                            shape=(self.lookback,self.n_features), dtype=np.float32)

        self.reset()



    def step(self, action):

        ## Normalise action space
        normalised_action = action / np.sum(np.abs(action))

        done = False
        # Rebalance portfolio at close using return of the next date
        next_day_log_return = self.pricedata[self.index,:]
        # transaction cost
        transaction_cost = self.transaction_cost(normalised_action,self.position_series[-1])

        # Rebalancing
        self.position_series = np.append(self.position_series, [normalised_action], axis=0)
        # Portfolio return
        today_portfolio_return = np.sum(normalised_action[:-1] * next_day_log_return) + np.sum(transaction_cost)
        self.log_return_series = np.append(self.log_return_series, [today_portfolio_return], axis=0)


        # Calculate reward
        # Need to cast log_return in pd series to use the functions in empyrical
        live_days = self.index - self.lookback
        burnin = 250
        recent_series = pd.Series(self.log_return_series)[-100:]
        whole_series = pd.Series(self.log_return_series)
        if live_days > burnin:
            self.metric = annual_return(whole_series) + 0.5* max_drawdown(whole_series)
        else:
            self.metric = annual_return(whole_series) + 0.5* max_drawdown(whole_series) *live_days / burnin
        reward = self.metric - self.metric_series[-1]
        #reward = self.metric
        self.metric_series = np.append(self.metric_series, [self.metric], axis=0)

        # Check if the end of backtest
        if self.index >= self.pricedata.shape[0]-2:
            done = True

        # Prepare observation for next day
        self.index += 1
        self.observation = self.get_observation()

        return self.observation, reward, done, {'current_price':next_day_log_return}


    def reset(self):
        self.log_return_series = np.zeros(shape=self.lookback)
        self.metric_series = np.zeros(shape=self.lookback)
        self.position_series = np.zeros(shape=(self.lookback,self.n_assets))
        self.metric = 0
        self.index = self.lookback
        self.observation = self.get_observation()
        return self.observation

    def get_observation(self):
        price_lookback = self.featuredata[self.index-self.lookback:self.index,:]
        metrics = np.vstack((self.log_return_series[self.index-self.lookback:self.index],
                             self.metric_series[self.index-self.lookback:self.index])).transpose()
        positions = self.position_series[self.index-self.lookback:self.index]
        observation = np.concatenate((price_lookback, metrics, positions), axis=1)
        return observation

    # 0.05% t-cost for institutional portfolios
    def transaction_cost(self,new_action,old_action,):
        turnover = np.abs(new_action - old_action)
        fees = 0.9995
        tcost = turnover * np.log(fees)
        return tcost

#set up the environment configs
my_config = {'pricing_source':'csvdata', 'tickers':['QQQ','EEM','TLT','SHY','GLD','SLV'], 'lookback':Lookback, 'start':'2010-01-02', 'end':'2018-12-31'}
EQ_env = Equitydaily(my_config)


#fire up Ray with the PPO agent

from ray.rllib.agents.ppo import PPOTrainer, DEFAULT_CONFIG
from ray.tune.logger import pretty_print

#set the PPO configs
config = DEFAULT_CONFIG
config['model']['dim'] = 50
config['model']['conv_filters'] = [[16, [5, 1], 1], [32, [5, 1], 5], [16, [10, 1], 1]]
config = DEFAULT_CONFIG.copy()
config['num_workers'] = 1
config["num_envs_per_worker"] = 1
config["rollout_fragment_length"] = 20
config["train_batch_size"] = 5000
config["batch_mode"] = "complete_episodes"
config['num_sgd_iter'] = 20
config['sgd_minibatch_size'] = 200
config['model']['fcnet_hiddens'] = [100, 100]
config['num_cpus_per_worker'] = 2  # This avoids running out of resources in the notebook environment when this cell is re-executed
config['env_config'] = my_config

#initialise the agent
agent = PPOTrainer(config, Equitydaily)
best_reward = -np.inf

print("Training")

#train all the agents for 1,000 iterations
for i in range(1000):
    result = agent.train()
    if result['episode_reward_mean'] > best_reward + 0.01:
        path = agent.save('sampleagent')
        #print(path)
        best_reward = result['episode_reward_mean']
        #print(best_reward)

#run the agent through the whole environment

state = EQ_env.reset()
done = False
reward_list = []
cum_reward = 0
actions = list()

while done == False:
    #action = agent.compute_action(state)
    #action = np.array([1,1,100,10,1000,1,1])
    action = agent.compute_action(state)
    state, reward, done, future_price = EQ_env.step(action)
    cum_reward += reward
    actions.append(action)
    reward_list.append(reward)

np.savetxt( "/rds/general/user/asm119/home/reinforcement_learning/Aidan/Lookback_sweep/LOOKBACK_" + str(Lookback) + ".txt", EQ_env.log_return_series)
