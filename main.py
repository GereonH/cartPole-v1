# Teil 1
import gym
import random
import numpy as np
import tflearn
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from statistics import median, mean
from collections import Counter

# Teil 2
LR = 1e-3
env = gym.make("CartPole-v1")
env.reset()
goal_steps = 500
durchläufe = int(input("Wieviele Durchläufe (default: 1): ") or 1)
if durchläufe == 1:
    score_requirement = int(input("Bitte geben Sie das Score Requirement an (default: 30): ") or 30)
initial_games = int(input("Bitte geben Sie die Anzahl der Trainingsspiele an (default: 10000): ") or 10000)
epochs = int(input("Über wieviele Epochen soll das Netz optimiert werden (default: 3): ") or 3)
plotting_initial = int(input("Soll die initiale Population geplottet werden? 1: Ja, 2: Nein (default) ") or 2)
plotting_final = int(input("Sollen die finalen Ergebnisse geplottet werden? 1: Ja, 2: Nein (default) ") or 2)
render = int(input("Sollen die Spiele gerendert werden? 1: Ja, 2: Nein(default) ") or 2)
if render == 1:
    render_games = True
else:
    render_games = False


# Teil 3
def initial_population(sc):
    # [OBS, MOVES]
    training_data = []
    # all scores:
    scores = []
    # just the scores that met our threshold:
    accepted_scores = []
    # iterate through however many games we want:
    for _ in range(initial_games):
        score = 0
        # moves specifically from this environment:
        game_memory = []
        # previous observation that we saw
        prev_observation = []
        # for each frame in 200
        for _ in range(goal_steps):
            # choose random action (0 or 1)
            action = random.randrange(0,2)
            # do it!
            observation, reward, done, info = env.step(action)

            # notice that the observation is returned FROM the action
            # so we'll store the previous observation here, pairing
            # the prev observation to the action we'll take.
            if len(prev_observation) > 0 :
                game_memory.append([prev_observation, action])
            prev_observation = observation
            score+=reward
            if done:
                break

        # IF our score is higher than our threshold, we'd like to save
        # every move we made
        # NOTE the reinforcement methodology here.
        # all we're doing is reinforcing the score, we're not trying
        # to influence the machine in any way as to HOW that score is
        # reached.
        if score >= sc:
            accepted_scores.append(score)
            for data in game_memory:
                # convert to one-hot (this is the output layer for our neural network)
                if data[1] == 1:
                    output = [0,1]
                elif data[1] == 0:
                    output = [1,0]

                # saving our training data
                training_data.append([data[0], output])

        # reset env to play again
        env.reset()
        # save overall scores
        scores.append(score)


    # just in case you wanted to reference later
    training_data_save = np.array(training_data)
    np.save('saved.npy',training_data_save)


    # plotting this stuff
    if plotting_initial == 1:
        # some stats here, to further illustrate the neural network magic!
        print('Average accepted score:',mean(accepted_scores))
        print('Median score for accepted scores:',median(accepted_scores))
        print('Counter accepted scores: ', Counter(accepted_scores))

        plt_data    = Counter(scores)
        plt_values  = list(plt_data.values())
        plt_count   = list(plt_data.keys())

        # Plot für die einzelnen Durchläufe
        plt.subplot(211)
        plt.plot(scores, 'ro')
        plt.axhline(y=sc, color='0')
        plt.xlabel('Episode')
        plt.ylabel('Score')
        plt.title(r'Score der einzelnen Episoden')

        # Histogramm für die Score-Häufigkeit
        plt.subplot(212)
        plt.bar(plt_count,plt_values)
        plt.axvline(x=sc, color='0')
        plt.xlabel('Score')
        plt.ylabel('Score')
        plt.title(r'Histogramm der Score-Häufigkeit')
        plt.suptitle('Resultat Zufallsbewegung')
        plt.show()

    return training_data

# Teil 4
def neural_network_model(input_size):

    network = input_data(shape=[None, input_size, 1], name='input')

    network = fully_connected(network, 128, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 256, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 512, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 256, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 128, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 2, activation='softmax')
    network = regression(network, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')
    model = tflearn.DNN(network)

    return model


# Teil 5
def train_model(training_data, model=False):

    X = np.array([i[0] for i in training_data]).reshape(-1,len(training_data[0][0]),1)
    y = [i[1] for i in training_data]

    # if not model:
    model = neural_network_model(input_size = len(X[0]))

    model.fit({'input': X}, {'targets': y}, n_epoch=epochs, snapshot_step=500, show_metric=True, run_id='openai_learning1')
    X = []
    y = []
    return model


# Teil 6
avgScores = []
scoreRequirements = []
for d in range(durchläufe):
    tf.reset_default_graph()
    file_avgScores = open("avgScores.txt","a")
    file_scoreRequirements = open("scoreRequirements.txt","a")
    print("Durchlauf: ", d+1)
    model = False

    if durchläufe == 1:
        training_data = initial_population(score_requirement)
    else:
        training_data = initial_population(d)
    print("Trainig Data abgeschlossen")

    model = train_model(training_data)
    print("Trainig Model abgeschlossen")

    scores = []
    choices = []

    for each_game in range(100):
        score = 0
        game_memory = []
        prev_obs = []
        env.reset()
        for _ in range(goal_steps):
            if render_games:
                env.render()

            if len(prev_obs)==0:
                action = random.randrange(0,2)
            else:
                action = np.argmax(model.predict(prev_obs.reshape(-1,len(prev_obs),1))[0])

            choices.append(action)

            new_observation, reward, done, info = env.step(action)
            prev_obs = new_observation
            game_memory.append([new_observation, action])
            score+=reward
            if done: break

        scores.append(score)

    avgScore = sum(scores)/len(scores)
    avgScores.append(avgScore)
    file_avgScores.write(" "+str(avgScore))
    if durchläufe == 1:
        scoreRequirements.append(score_requirement)
    else:
        scoreRequirements.append(d)
        file_scoreRequirements.write(" "+str(d))

    if plotting_final == 1:
        print('Average Score:',sum(scores)/len(scores))
        print('choice 1:{}  choice 0:{}'.format(choices.count(1)/len(choices),choices.count(0)/len(choices)))

        if durchläufe == 1:
            print('score requirement: ', score_requirement)
        else:
            print('score requirement: ', d)

        plt.plot(scores, 'ro')
        plt.axhline(y=(sum(scores)/len(scores)), color='tab:orange')
        if durchläufe == 1:
            plt.axhline(y=score_requirement, color='0')
        else:
            plt.axhline(y=d, color='0')
        plt.show()

print(avgScores)
print(scoreRequirements)

df = pd.DataFrame(avgScores)
rollingMean = df.rolling(window=5).mean()

plt.title("Score abhängig vom Score Requirement")
plt.plot(avgScores, 'b-', label="Durchschnitts-Score")
plt.plot(scoreRequirements, 'r-', label="Score Requirement")
plt.plot(rollingMean, 'g-',linewidth=4, label="Gleitender Mittelwert")
plt.xlabel('Score Requirement')
plt.ylabel('Durchschnitts-Score')
plt.legend()
plt.show()
