import gym
import random
import numpy as np
from keras.models     import Sequential
from keras.layers     import Dense
from keras.optimizers import Adam

env = gym.make('CartPole-v1')
env.reset()
goal_steps = 500 #massimi step di gioco utilizzabili
score_requirement = 50
intial_games = 50000 #numero di partite simulate


def model_data_preparation():
    training_data = []
    accepted_scores = []
    for game_index in range(intial_games): #gioco le partite (1000)
        score = 0
        game_memory = []
        previous_observation = []
        for step_index in range(goal_steps):#faccio al più 500 movimenti
            action = random.randrange(0, 2)
            observation, reward, done, info = env.step(action) # faccio un azione

            if len(previous_observation) > 0:
                game_memory.append([previous_observation, action]) #faccio un azione e vado in append (mi salvo il gioco)

            previous_observation = observation
            score += reward
            if done:
                break

        if score >= score_requirement: # se è almeno positivo lo registro come partita
            accepted_scores.append(score)
            for data in game_memory:
                if data[1] == 1:
                    output = [0, 1]
                elif data[1] == 0:
                    output = [1, 0]
                training_data.append([data[0], output])

        env.reset()

    print('Simulazione random Average Score:', sum(accepted_scores) / len(accepted_scores))
    print(accepted_scores)

    return training_data

training_data = model_data_preparation()

def build_model(input_size, output_size):
    model = Sequential()
    model.add(Dense(128, input_dim=input_size, activation='relu'))
    model.add(Dense(52, activation='relu'))
    model.add(Dense(output_size, activation='linear'))
    model.compile(loss='mse', optimizer=Adam())
    return model


def train_model(training_data):
    X = np.array([i[0] for i in training_data]).reshape(-1, len(training_data[0][0]))
    y = np.array([i[1] for i in training_data]).reshape(-1, len(training_data[0][1]))
    model = build_model(input_size=len(X[0]), output_size=len(y[0]))

    model.fit(X, y, epochs=50)
    return model

trained_model = train_model(training_data)

def Simulated_result():
    scores = []
    choices = []

    for each_game in range(100):
        score = 0
        prev_obs = []
        for step_index in range(goal_steps):
            # De-commenta la riga successiva per visualizzare il gioco
            #env.render()
            if len(prev_obs)  == 0:
                action = random.randrange(0, 2)
            else:
                action = np.argmax(trained_model.predict(prev_obs.reshape(-1, len(prev_obs)))[0])

            choices.append(action)
            new_observation, reward, done, info = env.step(action)
            prev_obs = new_observation
            score += reward
            if done:
                break

        env.reset()
        scores.append(score)
    return scores,choices

scores,choices=Simulated_result()

print(scores)
print('Simulazione 1 Average Score:', sum(scores) / len(scores))
print('Simulazione 1choice 1:{}  choice 0:{}'.format(choices.count(1) / len(choices), choices.count(0) / len(choices)))

