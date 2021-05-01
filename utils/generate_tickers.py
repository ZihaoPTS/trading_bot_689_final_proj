import logging
import random
import os
import numpy as np
import pandas as pd
from gym_environments.trading_environment import TradingEnvironment
from gym_environments.trading_environment_single_indicator import TradingEnvironment_si
from gym_environments.multi_trading_environment import Multi_TradingEnvironment
from 
import backtrader

def get_tickers_from_csv(filename):
    all_tickers = pd.read_csv(filename, usecols=['ACT Symbol'])
    all_tickers = all_tickers[~all_tickers['ACT Symbol'].str.contains('\$')]
    return all_tickers.values.flatten()

def generate_training_test_environments(filename, training_set_size=100, test_set_size=20, *_, seed=None):
    all_tickers = list(get_tickers_from_csv(filename))

    assert test_set_size < len(all_tickers), 'Test set size >= number of tickers'

    random.seed(seed)
    random.shuffle(all_tickers)
    def create_environment():
        while all_tickers:
            ticker = all_tickers.pop()
            try:
                env = TradingEnvironment(ticker)
                return env
            except Exception:
                pass

        # Ran out of candidates for environments
        raise IndexError('Not enough valid tickers in dataset to create environments')

    training_envs = [create_environment() for _ in range(training_set_size)]
    test_envs = [create_environment() for _ in range(test_set_size)]

    return training_envs, test_envs

def generate_training_test_environments_s_i(filename, training_set_size=100, test_set_size=20, *_, seed=None, indicator=None):
    all_tickers = list(get_tickers_from_csv(filename))

    assert test_set_size < len(all_tickers), 'Test set size >= number of tickers'

    random.seed(seed)
    random.shuffle(all_tickers)
    def create_environment():
        while all_tickers:
            ticker = all_tickers.pop()
            try:
                env = TradingEnvironment_si(ticker,indicator)
                return env
            except Exception:
                pass

        # Ran out of candidates for environments
        raise IndexError('Not enough valid tickers in dataset to create environments')

    training_envs = [create_environment() for _ in range(training_set_size)]
    test_envs = [create_environment() for _ in range(test_set_size)]

    return training_envs, test_envs


def generate_training_test_environments_multi(filename,tickes_size, training_set_size=100, test_set_size=20, *_, seed=None):
    all_tickers = list(get_tickers_from_csv(filename))

    assert test_set_size*tickes_size < len(all_tickers), 'Test set size >= number of tickers'

    random.seed(seed)
    random.shuffle(all_tickers)
    def create_environment():
        while all_tickers:
            ticker = []
            for i in range(tickes_size)
            ticker.append(all_tickers.pop())
            try:
                env = Multi_TradingEnvironment(ticker)
                return env
            except Exception:
                pass

        # Ran out of candidates for environments
        raise IndexError('Not enough valid tickers in dataset to create environments')

    training_envs = [create_environment() for _ in range(training_set_size)]
    test_envs = [create_environment() for _ in range(test_set_size)]

    return training_envs, test_envs