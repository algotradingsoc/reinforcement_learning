import gym
from gym import spaces
from empyrical import max_drawdown, alpha_beta, sharpe_ratio, annual_return

import pandas as pd
import numpy as np
import typing
from datetime import datetime

import ray
from ray.rllib.agents.ppo import PPOTrainer, DEFAULT_CONFIG

from ray import tune
from ray.tune.schedulers import AsyncHyperBandScheduler, ASHAScheduler

from empyrical import max_drawdown, alpha_beta, sharpe_ratio, annual_return
from sklearn.preprocessing import StandardScaler


def load_data(
    price_source: str,
    tickers: typing.List[str],
    start: datetime,
    end: datetime,
    features: typing.List[str],
    target: str,
):
    """Returned price data to use in gym environment"""
    # Load data
    # Each dataframe will have columns date and a collection of fields
    # TODO: DataLoader from mongoDB
    # Raw price from DB, forward impute on the trading days for missing date
    # calculate the features (log return, volatility)
    if price_source in ["OM"]:
        feature_df = []
        target_df = []
        for t in tickers:
            df1 = pd.read_csv("csvdata/OM/{}.csv".format(t))
            df1["date"] = pd.to_datetime(df1["date"])
            df1 = df1[(df1["date"] >= start) & (df1["date"] <= end)]
            df1.set_index("date", inplace=True)
            selected_features = features
            feature_df.append(df1[selected_features])
            target_df.append(df1[target])
            ref_df_columns = df1[selected_features].columns

    # assume all the price_df are aligned and cleaned in the DataLoader
    merged_feature = pd.concat(feature_df, axis=1, join="outer")
    merged_target = pd.concat(target_df, axis=1, join="outer")
    # Imputer missing values with zeros
    target_tensor = merged_target[target].fillna(0.0).values

    return {
        "dates": merged_feature.index,
        "fields": ref_df_columns,
        "features": merged_feature.fillna(0.0).values,
        "target": target_tensor,
    }


class Equitydaily(gym.Env):
    def __init__(self, env_config):

        self.tickers = env_config["tickers"]
        self.lookback = env_config["lookback"]
        self.random_start = env_config["random_start"]
        self.trading_days = env_config[
            "trading_days"
        ]  # Number of days the algorithm runs before resetting
        # Load price data, to be replaced by DataLoader class
        raw_data = load_data(
            env_config["pricing_source"],
            env_config["tickers"],
            env_config["start"],
            env_config["end"],
            env_config["features"],
            env_config["target"],
        )
        # Set the trading dates, features and price data
        self.dates = raw_data["dates"]
        self.fields = raw_data["fields"]
        self.targetdata = raw_data["target"]
        self.featuredata = raw_data["features"]

        # Set up historical actions and rewards
        self.n_assets = len(self.tickers) + 1
        self.n_metrics = 2
        self.n_assets_fields = len(self.fields)
        self.n_features = (
            self.n_assets_fields * len(self.tickers) + self.n_assets + self.n_metrics
        )  # reward function
        # self.n_features = self.n_assets_fields * len(self.tickers)

        # Set up action and observation space
        # The last asset is cash
        self.action_space = spaces.Box(
            low=-1, high=1, shape=(len(self.tickers) + 1,), dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.lookback, self.n_features, 1),
            dtype=np.float32,
        )

        self.reset()

    def step(self, action):

        # Normalise action space
        normalised_action = action / np.sum(np.abs(action))
        self.actions = normalised_action

        done = False
        # Rebalance portfolio at close using return of the next date
        next_day_log_return = self.targetdata[self.index, :]
        # transaction cost
        transaction_cost = self.transaction_cost(self.actions, self.position_series[-1])

        # Rebalancing
        self.position_series = np.append(self.position_series, [self.actions], axis=0)
        # Portfolio return
        today_portfolio_return = np.sum(
            self.actions[:-1] * next_day_log_return
        ) + np.sum(transaction_cost)
        self.log_return_series = np.append(
            self.log_return_series, [today_portfolio_return], axis=0
        )

        # Calculate reward
        # Need to cast log_return in pd series to use the functions in empyrical
        recent_series = pd.Series(self.log_return_series)[-100:]
        rolling_volatility = np.std(recent_series)
        if rolling_volatility > 0:
            self.metric = today_portfolio_return / rolling_volatility
        else:
            self.metric = 0
        reward = self.metric
        self.metric_series = np.append(self.metric_series, [self.metric], axis=0)

        # Check if the end of backtest
        if self.trading_days is None:
            done = self.index >= self.pricedata.shape[0] - 2
        else:
            done = (self.index - self.start_index) >= self.trading_days

        # Prepare observation for next day
        self.index += 1
        self.observation = self.get_observation()
        self.currentdate = self.dates[self.index - 1]

        return self.observation, reward, done, {}

    def reset(self):
        self.log_return_series = np.zeros(shape=self.lookback)
        self.metric_series = np.zeros(shape=self.lookback)
        self.position_series = np.zeros(shape=(self.lookback, self.n_assets))
        self.metric = 0
        if self.random_start:
            num_days = len(self.dates)
            self.start_index = np.random.randint(
                self.lookback, num_days - self.trading_days
            )
            self.index = self.start_index
        else:
            self.start_index = self.lookback
            self.index = self.lookback
        self.actions = np.zeros(shape=self.n_assets)
        self.observation = self.get_observation()
        self.currentdate = self.dates[self.index - 1]
        return self.observation

    def get_observation(self):
        # Can use simple moving average data here
        price_lookback = self.featuredata[self.index - self.lookback : self.index, :]
        metrics = np.vstack(
            (
                self.log_return_series[
                    self.index
                    - self.start_index : self.index
                    - self.start_index
                    + self.lookback
                ],
                self.metric_series[
                    self.index
                    - self.start_index : self.index
                    - self.start_index
                    + self.lookback
                ],
            )
        ).transpose()
        positions = self.position_series[
            self.index
            - self.start_index : self.index
            - self.start_index
            + self.lookback
        ]
        scaler = StandardScaler()
        price_lookback = (
            pd.DataFrame(scaler.fit_transform(price_lookback))
            .rolling(20, min_periods=1)
            .mean()
            .values
        )
        observation = np.concatenate((price_lookback, metrics, positions), axis=1)
        return observation.reshape((observation.shape[0], observation.shape[1], 1))

    # 0.05% and spread to model t-cost for institutional portfolios
    def transaction_cost(
        self,
        new_action,
        old_action,
    ):
        turnover = np.abs(new_action - old_action)
        fees = np.array([0.999 for i in range(self.n_assets)])
        tcost = turnover * np.log(fees)
        return tcost


def train_EQ_env(config):
    # create train environment

    agent = PPOTrainer(config, Equitydaily)
    best_reward = -np.inf

    while best_reward < config["best_reward"]:
        result = agent.train()
        # create test environment
        testconfig = config["env_config"].copy()
        testconfig["start"] = config["teststart"]
        testconfig["end"] = config["testend"]
        testenv = Equitydaily(testconfig)
        # run agent on test environment
        state = testenv.reset()
        done = False
        cum_reward = 0
        while not done:
            action = agent.compute_action(state)
            state, reward, done, future_price = EQ_env.step(action)
            cum_reward += reward
        if cum_reward > best_reward:
            path = agent.save(config["outputmodels"])
            best_reward = cum_reward
        tune.report(reward=cum_reward)


if __name__ == "__main__":

    import sys

    expt_no = sys.argv[1]

    ray.init(num_cpus=60, ignore_reinit_error=True, log_to_driver=False)

    envconfig = {
        "pricing_source": "OM",
        "tickers": [
            "14912310",
            "16676410",
        ],
        "lookback": 200,
        "start": "2010-01-02",
        "end": "2015-12-31",
        "random_start": True,
        "trading_days": 600,
    }

    envconfig["features"] = [
        "logreturn",
        "adjclose",
        "open_close",
        "high_close",
        "low_close",
    ]
    envconfig["target"] = "target_1"

    config = DEFAULT_CONFIG.copy()
    config["num_workers"] = 15
    config["num_envs_per_worker"] = 5
    config["rollout_fragment_length"] = 50
    config["train_batch_size"] = 25000
    config["batch_mode"] = "complete_episodes"
    config["num_sgd_iter"] = 20
    config["sgd_minibatch_size"] = 2000
    config["model"]["dim"] = 200
    config["model"]["conv_filters"] = [
        [16, [5, 1], 5],
        [16, [5, 1], 5],
        [16, [5, 1], 5],
    ]
    config["num_cpus_per_worker"] = 2
    config["env_config"] = envconfig
    config["teststart"] = "2016-01-02"
    config["testend"] = "2019-12-31"
    config["outputmodels"] = "sampleagent"

    scheduler = ASHAScheduler(
        max_t=200,
        grace_period=10,
    )

    analysis = tune.run(
        train_EQ_env,
        metric="reward",
        mode="max",
        resources_per_trial={"cpu": 30},  # You can add "gpu": 0.1 here
        config=config,
        scheduler=scheduler,
        verbose=0,
        num_samples=2,
        local_dir="ppoagent",
        stop={"reward": 100},
        checkpoint_at_end=True,
    )

    checkpoints = analysis.get_trial_checkpoints_paths(
        trial=analysis.get_best_trial("episode_reward_mean"),
        metric="episode_reward_mean",
    )

    print(checkpoints)