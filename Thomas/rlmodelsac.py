import gym
from gym import spaces
from empyrical import max_drawdown, alpha_beta, sharpe_ratio, annual_return

import pandas as pd
import numpy as np

import ray
from ray.rllib.agents.sac import SACTrainer, DEFAULT_CONFIG
from ray.tune.logger import pretty_print

# Start up Ray. This must be done before we instantiate any RL agents.
ray.init(num_cpus=30, ignore_reinit_error=True, log_to_driver=False)

config = DEFAULT_CONFIG.copy()
config["num_workers"] = 10
config["num_envs_per_worker"] = 10

config["rollout_fragment_length"] = 10
config["train_batch_size"] = 2500
config["timesteps_per_iteration"] = 10000
config["buffer_size"] = 20000
config["n_step"] = 10

config["Q_model"]["fcnet_hiddens"] = [50, 50]
config["policy_model"]["fcnet_hiddens"] = [50, 50]
config["num_cpus_per_worker"] = 2 
config["env_config"] = {
    "pricing_source": "csvdata",
    "tickers": ["SPY", "QQQ", "SHY", "GLD", "TLT", "LQD"],
    "lookback": 200,
    "start": "2008-01-02",
    "end": "2018-12-31",
}


def load_data(
    price_source="csvdata",
    tickers=["SPY", "QQQ"],
    start="2008-01-02",
    end="2010-01-02",
):
    """Returned price data to use in gym environment"""
    ## Load data

    if price_source in ["csvdata"]:
        price_df = []
        for t in tickers:
            df1 = (
                pd.read_csv("csvdata/{}.csv".format(t)).set_index("date").loc[start:end]
            )
            price_df.append(df1)
    ## Merge data
    ## Reference dataframe is taken from the first ticker read where the column labels are assumed to be the same
    if len(price_df) > 0:
        ref_df = price_df[0]
        ref_df_columns = price_df[0].columns
        for i in range(1, len(price_df)):
            ref_df = ref_df.merge(price_df[i], how="outer", on="date",)
        merged_df = ref_df.sort_values(by="date").fillna(0)
    ## Prepare price tensor for observation space
    price_tensor = np.zeros(
        shape=(merged_df.shape[0], len(ref_df_columns), len(price_df))
    )
    for count in range(len(price_df)):
        price_tensor[:, :, count] = merged_df.values[
            :, len(ref_df_columns) * count: len(ref_df_columns) * (count + 1)
        ]

    return {"dates": merged_df.index, "fields": ref_df_columns, "data": price_tensor}


class Equitydaily(gym.Env):
    def __init__(self, env_config):

        self.tickers = env_config["tickers"]
        self.lookback = env_config["lookback"]
        # Load price data
        price_data = load_data(
            env_config["pricing_source"],
            env_config["tickers"],
            env_config["start"],
            env_config["end"],
        )
        self.dates = price_data["dates"]
        self.fields = price_data["fields"]
        self.pricedata = price_data["data"]
        # Set up historical actions and rewards
        self.n_assets = len(self.tickers) + 1
        self.n_metrics = 2
        self.n_features = (
            len(self.fields) * len(self.tickers) + self.n_assets + self.n_metrics
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
        # Rebalance portfolio at open, use log return of open price in the following day
        next_day_log_return = self.pricedata[self.index + 1, 0, :]
        # transaction cost
        transaction_cost = self.transaction_cost(
            normalised_action, self.position_series[-1]
        )

        # Rebalancing
        self.position_series = np.append(
            self.position_series, [normalised_action], axis=0
        )
        today_portfolio_return = np.sum(
            normalised_action[:-1] * next_day_log_return
        ) + np.sum(transaction_cost)
        self.log_return_series = np.append(
            self.log_return_series, [today_portfolio_return], axis=0
        )

        # Calculate reward
        # Need to cast log_return in pd series to use the functions in empyrical
        live_days = self.index - self.lookback
        burnin = 250
        recent_series = pd.Series(self.log_return_series)[-100:]
        whole_series = pd.Series(self.log_return_series)
        if live_days > burnin:
            self.metric = annual_return(whole_series) + 0.5 * max_drawdown(whole_series)
        else:
            self.metric = (
                annual_return(whole_series)
                + 0.5 * max_drawdown(whole_series) * live_days / burnin
            )
        reward = self.metric - self.metric_series[-1]
        # reward = self.metric
        self.metric_series = np.append(self.metric_series, [self.metric], axis=0)

        # Check if the end of backtest
        if self.index >= self.pricedata.shape[0] - 2:
            done = True

        # Prepare observation for next day
        self.index += 1
        price_lookback = self.pricedata[
            self.index - self.lookback : self.index, :, :
        ].reshape(self.lookback, -1)
        metrics = np.vstack(
            (
                self.log_return_series[self.index - self.lookback : self.index],
                self.metric_series[self.index - self.lookback : self.index],
            )
        ).transpose()
        self.observation = np.concatenate(
            (
                price_lookback,
                metrics,
                self.position_series[self.index - self.lookback : self.index],
            ),
            axis=1,
        )

        return self.observation, reward, done, {}

    def reset(self):

        self.log_return_series = np.zeros(shape=self.lookback)
        self.metric_series = np.zeros(shape=self.lookback)
        self.position_series = np.zeros(shape=(self.lookback, self.n_assets))

        self.metric = 0
        self.index = self.lookback
        # Observation join the price, metric and position
        price_lookback = self.pricedata[: self.index, :, :].reshape(self.lookback, -1)
        metrics = np.vstack((self.log_return_series, self.metric_series)).transpose()
        self.observation = np.concatenate(
            (price_lookback, metrics, self.position_series), axis=1
        )

        return self.observation

    def transaction_cost(
        self, new_action, old_action,
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

