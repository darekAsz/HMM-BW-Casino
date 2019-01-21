# -*- coding:utf-8 -*-
# Filename: test_weather.py
# Author：hankcs
# Date: 2016-08-06 PM6:04
import numpy as np

import hmm
import random

states = ('Healthy', 'Fever')

observations = ('1', '2', '3', '4', '5', '6')

start_probability = {'Healthy': 0.6, 'Fever': 0.4}

transition_probability = {
    'Healthy': {'Healthy': 0.7, 'Fever': 0.3},
    'Fever': {'Healthy': 0.4, 'Fever': 0.6},
}

emission_probability = {
    'Healthy': {'1': 0.16, '2': 0.16, '3': 0.16, '4': 0.16, '5': 0.16, '6': 0.16},
    'Fever': {'1': 0.1, '2': 0.1, '3': 0.1, '4': 0.1, '5': 0.1, '6': 0.5},
}


def generate_index_map(lables):
    index_label = {}
    label_index = {}
    i = 0
    for l in lables:
        index_label[i] = l
        label_index[l] = i
        i += 1
    return label_index, index_label


states_label_index, states_index_label = generate_index_map(states)
observations_label_index, observations_index_label = generate_index_map(observations)


def convert_observations_to_index(observations, label_index):
    list = []
    for o in observations:
        list.append(label_index[o])
    return list


def convert_map_to_vector(map, label_index):
    v = np.empty(len(map), dtype=float)
    for e in map:
        v[label_index[e]] = map[e]
    return v


def convert_map_to_matrix(map, label_index1, label_index2):
    m = np.empty((len(label_index1), len(label_index2)), dtype=float)
    for line in map:
        for col in map[line]:
            m[label_index1[line]][label_index2[col]] = map[line][col]
    return m


A = convert_map_to_matrix(transition_probability, states_label_index, states_label_index)
# print A
B = convert_map_to_matrix(emission_probability, states_label_index, observations_label_index)
# print B
observations_index = convert_observations_to_index(observations, observations_label_index)
pi = convert_map_to_vector(start_probability, states_label_index)
# print pi

h = hmm.HMM(A, B, pi)

# print " " * 7, " ".join(("%10s" % observations_index_label[i]) for i in observations_index)
# for s in range(0, 2):
#    print "%7s: " % states_index_label[s] + " ".join("%10s" % ("%f" % v) for v in V[s])
# print '\nThe most possible states and probability are:'

# for s in ss:
#    print states_index_label[s],
# print p


A = np.array([[0.5, 0.5], [0.5, 0.5]])
B = np.array([[0.16, 0.16, 0.16, 0.16, 0.16, 0.16], [0.16, 0.16, 0.16, 0.16, 0.16, 0.16]])
pi = np.array([0.5, 0.5])

# run a baum_welch_train


# print observations_data
# print states_data
for i in range(100):
    observations_data, states_data = h.simulate(100)
    for j in range(100):
        print(j)
        rand = random.random()
        if rand <= 0.1:
            observations_data[j] = 0
        elif rand <= 0.15:
            observations_data[j] = 1
        elif rand <= 0.2:
            observations_data[j] = 2
        elif rand <= 0.4:
            observations_data[j] = 3
        elif rand <= 0.5:
            observations_data[j] = 4
        elif rand <= 1.0:
            observations_data[j] = 5
    # print observations_data
    guess = hmm.HMM(A, B, pi)
    A, B, pi = guess.baum_welch_train(observations_data)

print(B)
