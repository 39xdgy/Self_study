import gym, random
import numpy as np
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam
from collections import deque



def create_model(env):
    model = Sequential()
    model.add(Dense(24, input_dim = env.observation_shape.shape, activation = 'relu'))
    model.add(Dense(48, activation = 'relu'))
    model.add(Dense(24, activation = 'relu'))
    model.add(Dense(env.action_space.n)) # what choice should it make

    optimizer = Adam(lr = 0.01)
    
    model.compile(loss = 'mse', optimizer = optimizer)

    return model
    















env = gym.make('CartPole-v0')

training_round = 20
training_length = 500

step = []


#print(env.action_space.n)
#print(env.observation_space.low)

'''
for i_episode in range(20):
    observation = env.reset()
    for t in range(100):
        env.render()
        print(observation)
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
'''
