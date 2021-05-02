from itertools import count
import os
import random
import sys
import gym
import tensorflow as tf
from data_structures.replay_buffer import ReplayBuffer
from gym_environments.trading_environment_single_indicator import TradingEnvironment_si
from optimizers import DQN, load_dqn_model, create_lstm_network
from utils.generate_tickers import generate_training_test_environments_s_i
from utils.test_model import test_lstm_model


import backtrader.indicators as btind
import numpy as np
'ADX','TRIX','UO','RSI'
def BBPct(data):
    return btind.BollingerBandsPct(data)
def ADX(data):
    return btind.ADX(data)
def TRIX(data):
    return btind.TRIX(data)
def UO(data):
    return btind.UltimateOscillator(data)
def RSI(data):
    return btind.RSI_EMA(safediv=True)
def TSI(data):
    return btind.TSI(data)
def AroonUpDownOsc(data):
    return btind.AroonUpDownOsc(data)
def PPO(data):
    return btind.PPO(data)
def AO(data):
    return btind.AwesomeOscillator(data)
def CCI(data):
    return btind.CommodityChannelIndex(data)
def CrossOver(data):
    return btind.CrossOver(data)
def DemarkPivotPoint(data):
    return btind.DemarkPivotPoint
def DPO(data):
    return btind.DetrendedPriceOscillator(data)
def DMA(data):
    return btind.DicksonMovingAverageEnvelope(data)
def DMI(data):
    return btind.DirectionalMovementIndex(data)

def DEMAOsc(data):
    return btind.DoubleExponentialMovingAverageOscillator(data)
def FibonacciPivotPoint(data):
    return btind.FibonacciPivotPoint(data)
def HMAOsc(data):
    return btind.HullMovingAverageOscillator(data)
def Hurst(data):
    return btind.HurstExponent(data) #dont touch. need 2000+ sample to be stable
def KST(data):
    return btind.KnowSureThing(data)
def LAGF(data):
    return btind.LaguerreFilter(data)
def LRSI(data):
    return btind.LaguerreRSI(data)
def MinusDI(data):
    return btind.MinusDirectionalIndicator(data)
def MomentumOsc(data):
    return btind.MomentumOscillator(data)
def SMAOsc(data):
    return btind.MovingAverageSimpleOscillator(data)
def PSAR(data):
    return btind.ParabolicSAR(data)
def PGO(data):
    return btind.PrettyGoodOscillator(data)
def ROC100(data):
    return btind.RateOfChange100(data)

#i think all Osc can only go from -100 to 100, maybe passing observation space as limit might help their performance?

indicator_dict = {#'BBPct' : BBPct,
                  #'ADX' : ADX,
                  #'TRIX' : TRIX,
                  #'UO' : UO,
                  #'RSI' : RSI,
                  #'TSI' : TSI,
                  #'AroonUpDownOsc' : AroonUpDownOsc,
                  #'PPO' : PPO,
                  #'AO' : AO,
                  #'CCI' : CCI,
                  'CrossOver' : CrossOver,
                  'DemarkPivotPoint' : DemarkPivotPoint,
                  'DPO' : DPO,
                  'DMA' : DMA,
                  'DMI' : DMI
                  
                  
                  
                  
                  
                  #'Hurst' : Hurst #dont touch. need 2000+ sample to be stable

                    }
indicator_names = [k for k,v in indicator_dict.items()]
indicator_list = [v for k,v in indicator_dict.items()]
tf.get_logger().setLevel('WARNING')

# Param config
model_name = 'lstm_dqn'
num_training = 20
num_test = 5
lr = 0.0003
epsilon = 0.8
min_epsilon = 0.01
epsilon_decay = 0.99
gamma = 0.8
layer_nodes = [128, 256, 128]
time_steps = 4
replay_buffer_size = 1_000_000
batch_size = 100
batches_per_episode = 10
update_target_every = 5

indicator_avg_profits = []
for indicator in indicator_list:
    indicator_avg_profit = []

    replay_buffer = ReplayBuffer(replay_buffer_size)
    batches_until_target_update = update_target_every

    # Create training environments
    print('Creating training/test environments...')
    training_envs, test_envs = generate_training_test_environments_s_i('data/ticker_list/nyse-listed.csv', num_training, num_test, seed=0,indicator=indicator)
    print('Done setting up environments.')


    #print(f'Creating new model for {model_name}')
    #alwayse creat new model

    model = create_lstm_network(training_envs[0], lr=lr, layer_nodes=layer_nodes, time_steps=time_steps)
    target_model = create_lstm_network(training_envs[0], lr=lr, layer_nodes=layer_nodes, time_steps=time_steps)

    best_test_profit = -float('inf')

    # Train until stopped manually
    testing_it = 8

    for i in range(testing_it):
        # Create optimizers for each training environment
        optimizers = [DQN(env, model=model, target_model=target_model, epsilon=epsilon, gamma=gamma, time_steps=time_steps, replay_buffer=replay_buffer, update_target_every=update_target_every) for env in training_envs]

        # Generate episode for each optimizer, then run a batch update
        for j, optimizer in enumerate(optimizers):

            sys.stdout.write(f'\rGenerating episodes, current epsilon {epsilon}, iteration {i} ({j}/{len(optimizers)})...')
            sys.stdout.flush()
            optimizer.add_episode_to_replay_buffer()

            for _ in range(batches_per_episode):
                optimizer.train_minibatch(100)
                batches_until_target_update -= 1
                if batches_until_target_update == 0:
                    optimizer.update_target_model()
                    batches_until_target_update = update_target_every

            epsilon *= epsilon_decay
            epsilon = max(epsilon, min_epsilon)
        print()

        # Determine test performance
        print('Done training, testing...')
        test_profit = test_lstm_model(model, test_envs)
        indicator_avg_profit.append(test_profit)
    indicator_avg_profits.append([sum(indicator_avg_profit)/testing_it,indicator_avg_profit])
    printable = []
    for k,v in enumerate(indicator_avg_profits):
        printable.append({indicator_names[k]:v})
    with open("indicator_result.txt","w") as write_file:
        write_file.write(str(printable))

