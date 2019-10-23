import gym
import numpy as np
from keras.layers import Dense, Flatten
from keras.models import Sequential
from keras.optimizers import Adam
from rl.agents import SARSAAgent, DQNAgent
from rl.memory import SequentialMemory
from rl.policy import EpsGreedyQPolicy, LinearAnnealedPolicy
import random

env = gym.make('CartPole-v1')
env.reset()
def agent(states, actions):
    model = Sequential()
    model.add(Flatten(input_shape=(1, states)))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(52, activation='relu'))
    model.add(Dense(actions, activation='linear'))
    return model


model = agent(env.observation_space.shape[0], env.action_space.n)
STEPS_RICERCA=50000
memory = SequentialMemory(limit=10000, window_length=1)
policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1., value_min=.1, value_test=.05, nb_steps=1000)
dqn = DQNAgent(model=model, nb_actions=env.action_space.n, memory=memory, nb_steps_warmup=10,
               target_model_update=1e-2, policy=policy,enable_dueling_network=True, dueling_type='avg')
dqn.compile(Adam(lr=1e-3), metrics=['mae'])
dqn.fit(env, nb_steps=STEPS_RICERCA, visualize=False, verbose=1)

scores = dqn.test(env, nb_episodes=100, visualize=False)
print('Average score over 100 test games:{}'.format(np.mean(scores.history['episode_reward'])))
