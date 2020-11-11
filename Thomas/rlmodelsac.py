import gym
from gym import spaces
from empyrical import max_drawdown, alpha_beta, sharpe_ratio, annual_return

import pandas as pd
import numpy as np

import ray
from ray.rllib.agents.sac import SACTrainer, DEFAULT_CONFIG


# Start up Ray. This must be done before we instantiate any RL agents.
ray.init(num_cpus=30, ignore_reinit_error=True, log_to_driver=False)

config = DEFAULT_CONFIG.copy()
config["num_workers"] = 10
config["num_envs_per_worker"] = 20

config["rollout_fragment_length"] = 10
config["train_batch_size"] = 5000
config["timesteps_per_iteration"] = 10
config["buffer_size"] = 1000
config["n_step"] = 5
config["learning_starts"] = 50

config["Q_model"]["fcnet_hiddens"] = [100, 25]
config["policy_model"]["fcnet_hiddens"] = [100, 25]
config["num_cpus_per_worker"] = 2
config["env_config"] = {
    "pricing_source": "csvdata",
    "tickers": ["QQQ", "EEM", "TLT", "SHY", "GLD", "SLV"],
    "lookback": 1,
    "start": "2007-01-02",
    "end": "2015-12-31",
}


def load_data(
    price_source="csvdata", tickers=["EEM", "QQQ"], start="2008-01-02", end="2010-01-02"
):
    """Returned price data to use in gym environment"""
    # Load data
    # Each dataframe will have columns date and a collection of fields
    # TODO: DataLoader from mongoDB
    # Raw price from DB, forward impute on the trading days for missing date
    # calculate the features (log return, volatility)
    if price_source in ["csvdata"]:
        feature_df = []
        price_tensor = []
        for t in tickers:
            df1 = (
                pd.read_csv("csvdata/{}.csv".format(t)).set_index("date").loc[start:end]
            )
            feature_df.append(df1)
            price_tensor.append(
                df1["return"]
            )  # assumed to the be log return of the ref price
            ref_df_columns = df1.columns

    # assume all the price_df are aligned and cleaned in the DataLoader
    merged_df = pd.concat(feature_df, axis=1, join="outer")
    price_tensor = np.vstack(price_tensor).transpose()

    return {
        "dates": merged_df.index,
        "fields": ref_df_columns,
        "pricedata": price_tensor,
        "data": merged_df.values,
    }


class Equitydaily(gym.Env):
    def __init__(self, env_config):

        self.tickers = env_config["tickers"]
        self.lookback = env_config["lookback"]
        # Load price data, to be replaced by DataLoader class
        raw_data = load_data(
            env_config["pricing_source"],
            env_config["tickers"],
            env_config["start"],
            env_config["end"],
        )
        # Set the trading dates, features and price data
        self.dates = raw_data["dates"]
        self.fields = raw_data["fields"]
        self.pricedata = raw_data["pricedata"]
        self.featuredata = raw_data["data"]
        # Set up historical actions and rewards
        self.n_assets = len(self.tickers) + 1
        self.n_metrics = 2
        self.n_assets_fields = len(self.fields)
        self.n_features = (
            self.n_assets_fields * len(self.tickers) + self.n_assets + self.n_metrics
        )  # reward function

        # Set up action and observation space
        # The last asset is cash
        self.action_space = spaces.Box(
            low=-1, high=1, shape=(len(self.tickers) + 1,), dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.lookback, self.n_features),
            dtype=np.float32,
        )

        self.reset()

    def step(self, action):

        ## Normalise action space
        normalised_action = action / np.sum(np.abs(action))

        done = False
        # Rebalance portfolio at close using return of the next date
        next_day_log_return = self.pricedata[self.index, :]
        # transaction cost
        transaction_cost = self.transaction_cost(
            normalised_action, self.position_series[-1]
        )

        # Rebalancing
        self.position_series = np.append(
            self.position_series, [normalised_action], axis=0
        )
        # Portfolio return
        today_portfolio_return = np.sum(
            normalised_action[:-1] * next_day_log_return
        ) + np.sum(transaction_cost)
        self.log_return_series = np.append(
            self.log_return_series, [today_portfolio_return], axis=0
        )

        # Calculate reward
        # Need to cast log_return in pd series to use the functions in empyrical
        live_days = self.index - self.lookback
        burnin = 50
        recent_series = pd.Series(self.log_return_series)[-burnin:]
        whole_series = pd.Series(self.log_return_series)
        if live_days > burnin:
            self.metric = annual_return(recent_series) + 0.5 * max_drawdown(
                recent_series
            )
        else:
            self.metric = (
                (annual_return(whole_series) + 0.5 * max_drawdown(whole_series))
                * live_days
                / burnin
            )
        reward = self.metric - self.metric_series[-1]
        # reward = self.metric
        self.metric_series = np.append(self.metric_series, [self.metric], axis=0)

        # Check if the end of backtest
        if self.index >= self.pricedata.shape[0] - 2:
            done = True

        # Prepare observation for next day
        self.index += 1
        self.observation = self.get_observation()

        return self.observation, reward, done, {"current_price": next_day_log_return}

    def reset(self):
        self.log_return_series = np.zeros(shape=self.lookback)
        self.metric_series = np.zeros(shape=self.lookback)
        self.position_series = np.zeros(shape=(self.lookback, self.n_assets))
        self.metric = 0
        self.index = self.lookback
        self.observation = self.get_observation()
        return self.observation

    def get_observation(self):
        price_lookback = self.featuredata[self.index - self.lookback : self.index, :]
        metrics = np.vstack(
            (
                self.log_return_series[self.index - self.lookback : self.index],
                self.metric_series[self.index - self.lookback : self.index],
            )
        ).transpose()
        positions = self.position_series[self.index - self.lookback : self.index]
        observation = np.concatenate((price_lookback, metrics, positions), axis=1)
        return observation

    # 0.05% t-cost for institutional portfolios
    def transaction_cost(
        self,
        new_action,
        old_action,
    ):
        turnover = np.abs(new_action - old_action)
        fees = 0.9995
        tcost = turnover * np.log(fees)
        return tcost


# Train agent
agent = SACTrainer(config, Equitydaily)

best_reward = -0.4
for i in range(50000):
    result = agent.train()
    if (result["episode_reward_mean"] > best_reward + 0.01) or (i % 1000 == 500):
        path = agent.save("sacagent")
        print(path)
        if result["episode_reward_mean"] > best_reward + 0.01:
            best_reward = result["episode_reward_mean"]
            print(i, best_reward)
