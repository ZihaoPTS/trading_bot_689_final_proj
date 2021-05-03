import logging
import random
import os
import numpy as np
import pandas as pd
from gym_environments.trading_environment import TradingEnvironment
import backtrader
from datetime import datetime
default_fromdate = datetime(2018, 1, 1)
default_todate = datetime(2021, 4, 1)

def get_tickers_from_csv(filename):
    all_tickers = pd.read_csv(filename, usecols=['ACT Symbol'])
    all_tickers = all_tickers[~all_tickers['ACT Symbol'].str.contains('\$')]
    return all_tickers.values.flatten()

def generate_training_test_environments(filename, training_set_size=100, test_set_size=20, *_, seed=None,
    train_fromdate=default_fromdate, train_todate=default_todate, test_fromdate=default_fromdate, test_todate=default_todate):
    all_tickers = list(get_tickers_from_csv(filename))

    assert test_set_size < len(all_tickers), 'Test set size >= number of tickers'

    random.seed(seed)
    random.shuffle(all_tickers)
    def create_train_environment():
        while all_tickers:
            ticker = all_tickers.pop()
            try:
                env = TradingEnvironment(ticker,fromdate=train_fromdate,todate=train_todate)
                return env
            except Exception as e:
                print(e)
                pass

        # Ran out of candidates for environments
        raise IndexError('Not enough valid tickers in dataset to create environments')
    def create_test_environment():  
        while all_tickers:
            ticker = all_tickers.pop()
            try:
                env = TradingEnvironment(ticker,fromdate=test_fromdate,todate=test_todate)
                return env
            except Exception:
                pass
        raise IndexError('Not enough valid tickers in dataset to create environments')
    training_envs = [create_train_environment() for _ in range(training_set_size)]
    test_envs = [create_test_environment() for _ in range(test_set_size)]

    return training_envs, test_envs

#helper function for testing single indicator
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