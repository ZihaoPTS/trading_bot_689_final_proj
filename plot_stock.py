from collections import deque
import datetime
from gym_environments.trading_environment import TradingEnvironment
from gym_environments.trading_environment_static_boundary import TradingEnvironment_sb
import numpy as np
from matplotlib import pyplot as plt

import datetime
from matplotlib import pyplot
from collections import deque, namedtuple
import os
from typing import List
import numpy as np

from gym_environments.trading_environment import TradingState
from optimizers import load_a2c_model, load_a2c_lstm_model, load_dqn_model
from utils.generate_tickers import generate_training_test_environments

def plot_stock(model, ticker):
    state_size = model.input_shape[-1]
    num_timesteps = model.input_shape[-2]

    # Set initial value to 5 * stock value
    initial_money = 10_000
    env = TradingEnvironment_sb(ticker, initial_money=initial_money)
    s = env.reset()
    initial_stock_can_buy = 5
    initial_money = initial_stock_can_buy * s.price
    initial_price = s.price
    env = TradingEnvironment_sb(ticker, initial_money=initial_money)
    nA = env.action_space.n

    money = initial_money
    stock_owned = 0

    current_date = datetime.datetime(2018, 1, 1)

    stock_values = [s.price]
    portfolio_values = [initial_money]
    dates = [current_date]

    done = False
    steps = deque()
    while len(steps) < num_timesteps:
        steps.append(env.step(1)[0])
    steps = deque((0,) * state_size for _ in range(num_timesteps))
    while not done:
        steps.append(s)
        steps.popleft()

        # a = np.random.choice(nA, p=model(np.array([steps]))[0].numpy())
        # vals = model(np.array([steps])[0])
        # print(vals)
        a = np.argmax(model(np.array([steps]))[0])
        if a == 0 and money >= s.price:
            print(f'{current_date}: Buying stock at {s.price} (now {stock_owned + 1}x): ${money} -> ${money - s.price}')
            money -= s.price
            stock_owned += 1
        elif a == 2 and stock_owned > 0:
            print(f'{current_date}: Selling stock at {s.price}: ${money} -> ${money + s.price}')
            money += s.price * stock_owned
            stock_owned = 0

        s_prime, _, done, _ = env.step(a)

        s = s_prime
        current_date = current_date + datetime.timedelta(days=1)

        stock_values.append(s.price)
        portfolio_value = money + stock_owned * s.price
        portfolio_values.append(portfolio_value)
        dates.append(current_date)

    percent_profit = (portfolio_value - initial_money) / initial_money * 100
    portfolio_to_stock_gain_ratio = (portfolio_value - initial_money) / (s.price - initial_price) / initial_stock_can_buy
    print(f'Final value: {portfolio_value}\nPortfolio vs. stock change ratio: {portfolio_to_stock_gain_ratio}\nProfit: {percent_profit}%')
    plt.title(f'Stock and holding value over time for {ticker}')
    plt.plot_date(dates, stock_values, '-', color='orange')
    plt.plot_date(dates, portfolio_values, '-', color='blue')
    plt.show()

#model_name = 'lstm_dqn_recent'
model_name = 'lstm_dqn_sb'
model_is_lstm = True
model_is_dqn = True
num_stocks = 10
initial_money = 10_000
seed = 123123123
ticker_list_location = 'data/ticker_list/nyse-listed.csv'

model_save_location = os.path.join('models', model_name)

# Load model
try:
    if model_is_lstm:
        if model_is_dqn:
            model_save_location = os.path.join(model_save_location, 'model')
            model, _ = load_dqn_model(model_save_location, model_save_location)
        else:
            actor_save_location = os.path.join(model_save_location, 'actor')
            critic_save_location = os.path.join(model_save_location, 'critic')
            model, _ = load_a2c_lstm_model(actor_save_location, critic_save_location)
    else:
        model = load_a2c_model(model_save_location)
except Exception as e:
    print(f'Error loading model: {str(e)}')
    exit(1)

test = ['IMAX', 'M', 'SHLDQ', 'PCG', 'ANGI', 'ADMS', 'WBA','TSLA', 'AAPL', 'WMT', 'MSFT']
plot_stock(model,test[3])